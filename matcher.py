"""
matcher.py - The RAG matching engine.

Contains the CodeMatcher class that:
1. Retrieves top-K candidates from ChromaDB
2. Sends them to the LLM with the matching prompt
3. Parses the structured JSON response
"""

# Set environment variable BEFORE any TensorFlow imports to use legacy Keras
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
import time
import asyncio
import pandas as pd
from langchain_chroma import Chroma
import config
from prompts import MATCH_PROMPT, format_candidates


class CodeMatcher:
    """RAG-based code matching engine."""

    def __init__(self):
        """Load vector store and LLM based on configuration."""
        # Initialize embedding model using sentence-transformers (PyTorch-based, no Keras)
        from sentence_transformers import SentenceTransformer
        
        class SentenceTransformerEmbeddings:
            """Custom embedding class for LangChain compatibility."""
            def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)
            
            def embed_documents(self, texts: list) -> list:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            
            def embed_query(self, text: str) -> list:
                embedding = self.model.encode([text], convert_to_numpy=True)
                return embedding[0].tolist()
        
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load vector store from persistent directory
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
        )

        # Initialize LLM
        if config.USE_LOCAL:
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="gemma3:1b", temperature=config.LLM_TEMPERATURE)
            self.use_new_genai = False
        else:
            # Use the NEW google.genai SDK (the old one is deprecated)
            from google import genai
            self.genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
            self.llm = None  # We'll call the client directly
            self.use_new_genai = True

    def match_single(self, service_description: str) -> dict:
        """
        Match a single service description to a billing code.

        Args:
            service_description: The input service description to match.

        Returns:
            Dictionary with matched_code, code_system, matched_description,
            confidence, and reasoning.
        """
        # Handle empty or NaN service descriptions
        if not service_description or pd.isna(service_description):
            return {
                "matched_code": None,
                "code_system": None,
                "matched_description": None,
                "confidence": "NONE",
                "reasoning": "Empty or invalid service description",
            }

        service_description = str(service_description).strip()
        if not service_description or service_description.lower() == 'nan':
            return {
                "matched_code": None,
                "code_system": None,
                "matched_description": None,
                "confidence": "NONE",
                "reasoning": "Empty or invalid service description",
            }

        # Step 1: Retrieve top-K candidates with relevance scores
        results = self.vector_store.similarity_search_with_relevance_scores(
            service_description,
            k=config.TOP_K,
        )

        if not results:
            return {
                "matched_code": None,
                "code_system": None,
                "matched_description": None,
                "confidence": "NONE",
                "reasoning": "No candidates found in vector database",
            }

        # Extract documents and scores
        docs = [doc for doc, score in results]
        top_doc, top_score = results[0]

        # Step 2: Auto-accept optimization for high similarity matches
        if top_score > config.AUTO_ACCEPT_THRESHOLD:
            return {
                "matched_code": top_doc.metadata["code"],
                "code_system": top_doc.metadata["system"],
                "matched_description": top_doc.metadata["description"],
                "confidence": "HIGH",
                "reasoning": f"Auto-matched (similarity={top_score:.3f})",
            }

        # Step 3: Build prompt with candidates
        candidates_text = format_candidates(docs)
        prompt = MATCH_PROMPT.format(
            service_description=service_description,
            candidates=candidates_text,
        )

        # Step 4: Call LLM
        try:
            if config.USE_LOCAL:
                response_text = self.llm.invoke(prompt)
            elif self.use_new_genai:
                # Use the new google.genai SDK
                response = self.genai_client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=prompt
                )
                response_text = response.text
            else:
                response = self.llm.invoke(prompt)
                response_text = response.content
        except Exception as e:
            # If LLM call fails, fall back to top retrieval result
            return {
                "matched_code": top_doc.metadata["code"],
                "code_system": top_doc.metadata["system"],
                "matched_description": top_doc.metadata["description"],
                "confidence": "LOW",
                "reasoning": f"LLM call failed: {str(e)[:50]}; using vector similarity",
            }

        # Step 5: Parse JSON response
        result = self._parse_llm_response(response_text, top_doc)
        return result

    def _parse_llm_response(self, response_text: str, fallback_doc) -> dict:
        """
        Parse the LLM's JSON response with fallback handling.

        Args:
            response_text: The raw text response from the LLM.
            fallback_doc: The top retrieval document to use as fallback.

        Returns:
            Parsed result dictionary.
        """
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            pass

        # Fallback: try stripping markdown code fences
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError:
            pass

        # Final fallback: use top retrieval result with LOW confidence
        return {
            "matched_code": fallback_doc.metadata["code"],
            "code_system": fallback_doc.metadata["system"],
            "matched_description": fallback_doc.metadata["description"],
            "confidence": "LOW",
            "reasoning": "LLM response parsing failed; using vector similarity",
        }

    def match_batch(self, input_excel_path: str, output_excel_path: str) -> list:
        """
        Process an entire mapping sheet.

        Args:
            input_excel_path: Path to the input Excel file with Service Code
                              and Service Description columns.
            output_excel_path: Path where the output Excel will be saved.

        Returns:
            List of tuples (sheet_name, DataFrame) with results.
        """
        import os
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        xls = pd.ExcelFile(input_excel_path)
        all_results = []
        stats = {}

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Strip column names to handle trailing spaces
            df.columns = df.columns.str.strip()

            # Only keep Service Code and Service Description
            if "Service Code" not in df.columns or "Service Description" not in df.columns:
                print(f"Warning: Sheet '{sheet_name}' missing required columns, skipping")
                continue

            print(f"\nProcessing sheet: {sheet_name} ({len(df)} rows)")

            matched_codes = []
            code_systems = []
            matched_descriptions = []
            confidences = []
            reasonings = []

            # Initialize stats for this sheet
            sheet_stats = {"total": len(df), "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}

            for idx, row in df.iterrows():
                desc = row.get("Service Description", "")
                desc_preview = str(desc)[:60] if desc else "(empty)"
                print(f"  [{sheet_name}] [{idx+1}/{len(df)}] {desc_preview}...")

                result = self.match_single(desc)

                matched_codes.append(result.get("matched_code", ""))
                code_systems.append(result.get("code_system", ""))
                matched_descriptions.append(result.get("matched_description", ""))
                confidences.append(result.get("confidence", "NONE"))
                reasonings.append(result.get("reasoning", ""))

                # Update stats
                confidence = result.get("confidence", "NONE")
                if confidence in sheet_stats:
                    sheet_stats[confidence] += 1

                # Rate limiting to avoid API throttling
                time.sleep(0.1)

            # Add new columns to DataFrame
            df["Matched Code"] = matched_codes
            df["Code System"] = code_systems
            df["Matched Description"] = matched_descriptions
            df["Confidence"] = confidences
            df["Reasoning"] = reasonings

            # Drop the empty NPHIES columns if they exist
            cols_to_drop = ["NPHIES Code", "Description", "Other Code Value"]
            for col in cols_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])

            all_results.append((sheet_name, df))
            stats[sheet_name] = sheet_stats

        # Write all sheets to output Excel
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            for sheet_name, df in all_results:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"\nResults saved to {output_excel_path}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("MATCHING SUMMARY")
        print("=" * 60)
        for sheet_name, s in stats.items():
            print(f"\n{sheet_name}:")
            print(f"  Total: {s['total']}")
            print(f"  HIGH:   {s['HIGH']} ({100*s['HIGH']/s['total']:.1f}%)")
            print(f"  MEDIUM: {s['MEDIUM']} ({100*s['MEDIUM']/s['total']:.1f}%)")
            print(f"  LOW:    {s['LOW']} ({100*s['LOW']/s['total']:.1f}%)")
            print(f"  NONE:   {s['NONE']} ({100*s['NONE']/s['total']:.1f}%)")

        return all_results

    async def match_batch_async(self, input_excel_path: str, output_excel_path: str) -> list:
        """
        Process an entire mapping sheet using async for faster processing.

        Args:
            input_excel_path: Path to the input Excel file.
            output_excel_path: Path where the output Excel will be saved.

        Returns:
            List of tuples (sheet_name, DataFrame) with results.
        """
        import os
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        xls = pd.ExcelFile(input_excel_path)
        all_results = []
        stats = {}

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(10)

        async def match_single_async(desc: str) -> dict:
            """Async wrapper for match_single."""
            async with semaphore:
                # Run the synchronous match in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.match_single, desc)
                return result

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()

            if "Service Code" not in df.columns or "Service Description" not in df.columns:
                print(f"Warning: Sheet '{sheet_name}' missing required columns, skipping")
                continue

            print(f"\nProcessing sheet: {sheet_name} ({len(df)} rows)")

            # Get all descriptions
            descriptions = df["Service Description"].tolist()

            # Process all descriptions concurrently
            tasks = [match_single_async(desc) for desc in descriptions]
            results = await asyncio.gather(*tasks)

            # Extract results into lists
            matched_codes = [r.get("matched_code", "") for r in results]
            code_systems = [r.get("code_system", "") for r in results]
            matched_descriptions = [r.get("matched_description", "") for r in results]
            confidences = [r.get("confidence", "NONE") for r in results]
            reasonings = [r.get("reasoning", "") for r in results]

            # Calculate stats
            sheet_stats = {"total": len(df), "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
            for conf in confidences:
                if conf in sheet_stats:
                    sheet_stats[conf] += 1

            # Add new columns to DataFrame
            df["Matched Code"] = matched_codes
            df["Code System"] = code_systems
            df["Matched Description"] = matched_descriptions
            df["Confidence"] = confidences
            df["Reasoning"] = reasonings

            # Drop empty NPHIES columns
            cols_to_drop = ["NPHIES Code", "Description", "Other Code Value"]
            for col in cols_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])

            all_results.append((sheet_name, df))
            stats[sheet_name] = sheet_stats

            print(f"  Completed {len(df)} rows")

        # Write all sheets to output Excel
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            for sheet_name, df in all_results:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"\nResults saved to {output_excel_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("MATCHING SUMMARY")
        print("=" * 60)
        for sheet_name, s in stats.items():
            print(f"\n{sheet_name}:")
            print(f"  Total: {s['total']}")
            print(f"  HIGH:   {s['HIGH']} ({100*s['HIGH']/s['total']:.1f}%)")
            print(f"  MEDIUM: {s['MEDIUM']} ({100*s['MEDIUM']/s['total']:.1f}%)")
            print(f"  LOW:    {s['LOW']} ({100*s['LOW']/s['total']:.1f}%)")
            print(f"  NONE:   {s['NONE']} ({100*s['NONE']/s['total']:.1f}%)")

        return all_results


if __name__ == "__main__":
    # Quick test
    print("Initializing CodeMatcher...")
    matcher = CodeMatcher()

    print("\nTesting single match: 'Cisternal puncture'")
    result = matcher.match_single("Cisternal puncture")
    print(json.dumps(result, indent=2))

    print("\nTesting single match: 'BEPANTHEN'")
    result = matcher.match_single("BEPANTHEN")
    print(json.dumps(result, indent=2))

    print("\nTesting single match: 'vitamin D3 IVD calibrator'")
    result = matcher.match_single("vitamin D3 IVD calibrator")
    print(json.dumps(result, indent=2))
