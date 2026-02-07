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
            self.llm = Ollama(model="gemma2:2b", temperature=config.LLM_TEMPERATURE)
        else:
            # Use AWS Bedrock
            from langchain_aws import ChatBedrock
            self.llm = ChatBedrock(
                model_id=config.LLM_MODEL,
                model_kwargs={"temperature": config.LLM_TEMPERATURE},
                region_name=config.AWS_REGION,
                credentials_profile_name=None,  # Use environment variables
            )
        
        # Initialize cache for performance optimization
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _extract_metadata(self, doc) -> dict:
        """Extract system-specific metadata from a document."""
        metadata = doc.metadata
        system = metadata.get("system", "")
        
        result = {
            "matched_code": metadata.get("code", ""),
            "code_system": system,
            "matched_description": metadata.get("description", ""),
        }
        
        # For SBS, add additional fields
        if system == "SBS":
            result["sbs_code_numeric"] = metadata.get("code_numeric", "")
            result["sbs_code_hyphenated"] = metadata.get("code_hyphenated", "")
            result["short_description"] = metadata.get("description", "")
        
        # For GTIN, add additional fields
        elif system == "GTIN":
            result["gtin_code"] = metadata.get("code", "")
            result["gtin_ingredients"] = metadata.get("ingredients", "")
            result["gtin_strength"] = metadata.get("strength", "")
        
        # For GMDN, add additional fields
        elif system == "GMDN":
            result["gmdn_code"] = metadata.get("code", "")
            result["gmdn_name"] = metadata.get("term_name", "")
            result["gmdn_definition"] = metadata.get("term_definition", "")
        
        return result
    
    def clear_cache(self):
        """Clear the match cache."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }

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

        # Check cache first for performance optimization
        cache_key = service_description.strip().lower()
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key].copy()
        
        self.cache_misses += 1

        # Step 1: Retrieve top-K candidates
        # Use similarity_search instead of similarity_search_with_relevance_scores
        # to avoid negative score warnings from ChromaDB
        docs = self.vector_store.similarity_search(
            service_description,
            k=config.TOP_K,
        )

        if not docs:
            return {
                "matched_code": None,
                "code_system": None,
                "matched_description": None,
                "confidence": "NONE",
                "reasoning": "No candidates found in vector database",
            }

        # Get top document for auto-accept check
        top_doc = docs[0]
        
        # For auto-accept, we'll use a simple heuristic based on exact text match
        # instead of relying on potentially negative similarity scores
        top_content = top_doc.page_content.lower()
        query_lower = service_description.lower()
        
        # Check if query is contained in top result or vice versa
        is_exact_match = (query_lower in top_content) or (top_content in query_lower)

        # Step 2: Auto-accept optimization for high similarity matches
        if is_exact_match:  # Using is_exact_match as a proxy for high confidence
            result = self._extract_metadata(top_doc)
            result["confidence"] = "HIGH"
            result["reasoning"] = "Auto-matched (exact text match)"
            self.cache[cache_key] = result  # Cache auto-accept results
            return result

        # Step 3: Build prompt with candidates
        candidates_text = format_candidates(docs)
        prompt = MATCH_PROMPT.format(
            service_description=service_description,
            candidates=candidates_text,
        )

        # Step 4: Call LLM with retry logic for throttling
        max_retries = 5
        base_delay = 5  # seconds (increased for more conservative rate limiting)
        
        response_text = None
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                # Extract text from response
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a throttling error
                if "ThrottlingException" in error_msg or "Too many requests" in error_msg:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limited, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Max retries reached for throttling")
                        return {
                            "matched_code": None,
                            "code_system": None,
                            "matched_description": None,
                            "confidence": "NONE",
                            "reasoning": "Rate limit exceeded, please try again later",
                        }
                else:
                    # Non-throttling error
                    print(f"LLM Error: {error_msg}")
                    # Print full error for debugging
                    import traceback
                    traceback.print_exc()
                    
                    # If LLM call fails, fall back to top retrieval result
                    result = self._extract_metadata(top_doc)
                    result["confidence"] = "LOW"
                    result["reasoning"] = f"LLM call failed: {error_msg[:100]}; using vector similarity"
                    return result
        
        # If response_text is still None after the loop, it means all retries failed
        # and the last error was not a throttling error that returned early.
        # This case should ideally be caught by the 'else' block above, but as a safeguard:
        if response_text is None:
            print(f"  ❌ LLM call failed after all retries (unknown reason).")
            result = self._extract_metadata(top_doc)
            result["confidence"] = "LOW"
            result["reasoning"] = "LLM call failed after retries; using vector similarity"
            return result

        # Step 5: Parse JSON response and enrich with metadata
        result = self._parse_llm_response(response_text, top_doc, docs)
        
        # Store in cache for future requests
        self.cache[cache_key] = result
        
        return result

    def _parse_llm_response(self, response_text: str, fallback_doc, candidates: list = None) -> dict:
        """
        Parse the LLM's JSON response with fallback handling.

        Args:
            response_text: The raw text response from the LLM.
            fallback_doc: The top retrieval document to use as fallback.
            candidates: List of candidate documents retrieved from vector store.

        Returns:
            Parsed result dictionary enriched with system-specific fields.
        """
        result = None
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
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
            except json.JSONDecodeError:
                pass

        # If parsing failed, use fallback
        if result is None:
            result = self._extract_metadata(fallback_doc)
            result["confidence"] = "LOW"
            result["reasoning"] = "LLM response parsing failed; using vector similarity"
            return result
        
        # Enrich the LLM result with system-specific fields by matching against candidates
        if candidates and result.get("matched_code"):
            matched_code = result.get("matched_code")
            code_system = result.get("code_system")
            
            # Find the matching document in candidates
            for doc in candidates:
                if (doc.metadata.get("code") == matched_code and 
                    doc.metadata.get("system") == code_system):
                    # Enrich with system-specific fields
                    if code_system == "SBS":
                        result["sbs_code_numeric"] = doc.metadata.get("code_numeric", "")
                        result["sbs_code_hyphenated"] = doc.metadata.get("code_hyphenated", "")
                        result["short_description"] = doc.metadata.get("description", "")
                    elif code_system == "GTIN":
                        result["gtin_code"] = doc.metadata.get("code", "")
                        result["gtin_ingredients"] = doc.metadata.get("ingredients", "")
                        result["gtin_strength"] = doc.metadata.get("strength", "")
                    elif code_system == "GMDN":
                        result["gmdn_code"] = doc.metadata.get("code", "")
                        result["gmdn_name"] = doc.metadata.get("term_name", "")
                        result["gmdn_definition"] = doc.metadata.get("term_definition", "")
                    break
        
        return result

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
            
            # SBS-specific fields
            sbs_code_numeric_list = []
            sbs_short_desc_list = []
            sbs_code_hyphenated_list = []
            
            # GTIN-specific fields
            gtin_code_list = []
            gtin_ingredients_list = []
            gtin_strength_list = []
            
            # GMDN-specific fields
        # GMDN-specific fields
            gmdn_code_list = []
            gmdn_name_list = []
            gmdn_definition_list = []

            # Initialize stats for this sheet
            sheet_stats = {"total": len(df), "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}

            # Process each row
            import time
            start_time = time.time()
            
            for idx, row in df.iterrows():
                desc = row.get("Service Description", "")
                desc_preview = str(desc)[:50] if desc else "(empty)"
                
                # Calculate progress
                progress = ((idx + 1) / len(df)) * 100
                elapsed = time.time() - start_time
                
                # Estimate time remaining
                if idx > 0:
                    avg_time_per_row = elapsed / (idx + 1)
                    remaining_rows = len(df) - (idx + 1)
                    eta_seconds = avg_time_per_row * remaining_rows
                    eta_minutes = eta_seconds / 60
                    
                    print(f"  [{sheet_name}] [{idx+1}/{len(df)}] ({progress:.1f}%) | ETA: {eta_minutes:.1f}m | {desc_preview}...")
                else:
                    print(f"  [{sheet_name}] [{idx+1}/{len(df)}] ({progress:.1f}%) | {desc_preview}...")

                result = self.match_single(desc)

                matched_codes.append(result.get("matched_code", ""))
                code_systems.append(result.get("code_system", ""))
                matched_descriptions.append(result.get("matched_description", ""))
                confidences.append(result.get("confidence", "NONE"))
                reasonings.append(result.get("reasoning", ""))
                
                # Collect SBS-specific fields
                sbs_code_numeric_list.append(result.get("sbs_code_numeric", ""))
                sbs_short_desc_list.append(result.get("short_description", ""))
                sbs_code_hyphenated_list.append(result.get("sbs_code_hyphenated", ""))
                
                # Collect GTIN-specific fields
                gtin_code_list.append(result.get("gtin_code", ""))
                gtin_ingredients_list.append(result.get("gtin_ingredients", ""))
                gtin_strength_list.append(result.get("gtin_strength", ""))
                
                # Collect GMDN-specific fields
                gmdn_code_list.append(result.get("gmdn_code", ""))
                gmdn_name_list.append(result.get("gmdn_name", ""))
                gmdn_definition_list.append(result.get("gmdn_definition", ""))

                # Update stats
                confidence = result.get("confidence", "NONE")
                if confidence in sheet_stats:
                    sheet_stats[confidence] += 1

                # Rate limiting to avoid API throttling
                time.sleep(0.1)

            # Add new columns to DataFrame in the requested order
            # For SBS: SBS Code (numeric), Short Description, SBS Code Hyphenated
            df["SBS Code"] = sbs_code_numeric_list
            df["Short Description"] = sbs_short_desc_list
            df["SBS Code Hyphenated"] = sbs_code_hyphenated_list
            
            # For GTIN: GTIN Code, Ingredients, Strength
            df["GTIN Code"] = gtin_code_list
            df["GTIN Ingredients"] = gtin_ingredients_list
            df["GTIN Strength"] = gtin_strength_list
            
            # For GMDN: GMDN Code, Name, Definition
            df["GMDN Code"] = gmdn_code_list
            df["GMDN Name"] = gmdn_name_list
            df["GMDN Definition"] = gmdn_definition_list
            
            # Then add the common columns
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

        # Combine all results and separate by code system
        from openpyxl.styles import PatternFill
        
        # Combine all dataframes
        all_dfs = [df for _, df in all_results]
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Separate by code system
        df_sbs = combined_df[combined_df["Code System"] == "SBS"].copy()
        df_gtin = combined_df[combined_df["Code System"] == "GTIN"].copy()
        df_gmdn = combined_df[combined_df["Code System"] == "GMDN"].copy()
        
        # Yellow fill for highlighting
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        
        # Write to single Excel file with 3 sheets
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            # Write SBS sheet
            if not df_sbs.empty:
                sbs_cols = ["Service Code", "Service Description", 
                           "SBS Code", "Short Description", "SBS Code Hyphenated",
                           "Confidence", "Reasoning"]
                df_sbs_output = df_sbs[[col for col in sbs_cols if col in df_sbs.columns]]
                df_sbs_output.to_excel(writer, sheet_name="SBS", index=False)
                
                # Apply yellow highlighting
                ws = writer.sheets["SBS"]
                for row in range(2, len(df_sbs_output) + 2):
                    ws.cell(row=row, column=3).fill = yellow_fill  # SBS Code
                    ws.cell(row=row, column=4).fill = yellow_fill  # Short Description
                    ws.cell(row=row, column=5).fill = yellow_fill  # SBS Code Hyphenated
            
            # Write GTIN sheet
            if not df_gtin.empty:
                gtin_cols = ["Service Code", "Service Description",
                            "GTIN Code", "GTIN Ingredients", "GTIN Strength",
                            "Confidence", "Reasoning"]
                df_gtin_output = df_gtin[[col for col in gtin_cols if col in df_gtin.columns]]
                df_gtin_output.to_excel(writer, sheet_name="GTIN", index=False)
                
                # Apply yellow highlighting
                ws = writer.sheets["GTIN"]
                for row in range(2, len(df_gtin_output) + 2):
                    ws.cell(row=row, column=3).fill = yellow_fill  # GTIN Code
                    ws.cell(row=row, column=4).fill = yellow_fill  # GTIN Ingredients
                    ws.cell(row=row, column=5).fill = yellow_fill  # GTIN Strength
            
            # Write GMDN sheet
            if not df_gmdn.empty:
                gmdn_cols = ["Service Code", "Service Description",
                            "GMDN Code", "GMDN Name", "GMDN Definition",
                            "Confidence", "Reasoning"]
                df_gmdn_output = df_gmdn[[col for col in gmdn_cols if col in df_gmdn.columns]]
                df_gmdn_output.to_excel(writer, sheet_name="GMDN", index=False)
                
                # Apply yellow highlighting
                ws = writer.sheets["GMDN"]
                for row in range(2, len(df_gmdn_output) + 2):
                    ws.cell(row=row, column=3).fill = yellow_fill  # GMDN Code
                    ws.cell(row=row, column=4).fill = yellow_fill  # GMDN Name
                    ws.cell(row=row, column=5).fill = yellow_fill  # GMDN Definition

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

        # Semaphore to limit concurrent requests (reduced to avoid rate limiting)
        semaphore = asyncio.Semaphore(2)  # Only 2 concurrent requests to AWS Bedrock

        async def match_single_async(desc: str) -> dict:
            """Async wrapper for match_single with rate limiting."""
            async with semaphore:
                # Run the synchronous match in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.match_single, desc)
                # Delay to avoid overwhelming the API
                await asyncio.sleep(1.0)  # 1 second delay between requests
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

            # Process all descriptions concurrently with progress tracking
            import time
            start_time = time.time()
            completed = 0
            total = len(descriptions)
            
            async def track_progress(task, idx):
                nonlocal completed
                result = await task
                completed += 1
                
                # Print progress every 10 rows or at milestones
                if completed % 10 == 0 or completed == total:
                    progress = (completed / total) * 100
                    elapsed = time.time() - start_time
                    
                    if completed > 1:
                        avg_time = elapsed / completed
                        remaining = total - completed
                        eta_seconds = avg_time * remaining
                        eta_minutes = eta_seconds / 60
                        print(f"  Progress: {completed}/{total} ({progress:.1f}%) | ETA: {eta_minutes:.1f}m")
                    else:
                        print(f"  Progress: {completed}/{total} ({progress:.1f}%)")
                
                return result
            
            tasks = [track_progress(match_single_async(desc), idx) for idx, desc in enumerate(descriptions)]
            results = await asyncio.gather(*tasks)

            # Extract results into lists
            matched_codes = [r.get("matched_code", "") for r in results]
            code_systems = [r.get("code_system", "") for r in results]
            matched_descriptions = [r.get("matched_description", "") for r in results]
            confidences = [r.get("confidence", "NONE") for r in results]
            reasonings = [r.get("reasoning", "") for r in results]
            
            # Extract SBS-specific fields
            sbs_code_numeric_list = [r.get("sbs_code_numeric", "") for r in results]
            sbs_short_desc_list = [r.get("short_description", "") for r in results]
            sbs_code_hyphenated_list = [r.get("sbs_code_hyphenated", "") for r in results]
            
            # Extract GTIN-specific fields
            gtin_code_list = [r.get("gtin_code", "") for r in results]
            gtin_ingredients_list = [r.get("gtin_ingredients", "") for r in results]
            gtin_strength_list = [r.get("gtin_strength", "") for r in results]
            
            # Extract GMDN-specific fields
            gmdn_code_list = [r.get("gmdn_code", "") for r in results]
            gmdn_name_list = [r.get("gmdn_name", "") for r in results]
            gmdn_definition_list = [r.get("gmdn_definition", "") for r in results]

            # Calculate stats
            sheet_stats = {"total": len(df), "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
            for conf in confidences:
                if conf in sheet_stats:
                    sheet_stats[conf] += 1

            # Add new columns to DataFrame in the requested order
            df["SBS Code"] = sbs_code_numeric_list
            df["Short Description"] = sbs_short_desc_list
            df["SBS Code Hyphenated"] = sbs_code_hyphenated_list
            
            df["GTIN Code"] = gtin_code_list
            df["GTIN Ingredients"] = gtin_ingredients_list
            df["GTIN Strength"] = gtin_strength_list
            
            df["GMDN Code"] = gmdn_code_list
            df["GMDN Name"] = gmdn_name_list
            df["GMDN Definition"] = gmdn_definition_list
            
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

        # Combine all results and separate by code system
        from openpyxl.styles import PatternFill
        
        # Combine all dataframes
        all_dfs = [df for _, df in all_results]
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Separate by code system
        df_sbs = combined_df[combined_df["Code System"] == "SBS"].copy()
        df_gtin = combined_df[combined_df["Code System"] == "GTIN"].copy()
        df_gmdn = combined_df[combined_df["Code System"] == "GMDN"].copy()
        
        # Yellow fill for highlighting
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        
        # Write to single Excel file with 3 sheets
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            # Write SBS sheet
            if not df_sbs.empty:
                sbs_cols = ["Service Code", "Service Description", 
                           "SBS Code", "Short Description", "SBS Code Hyphenated",
                           "Confidence", "Reasoning"]
                df_sbs_output = df_sbs[[col for col in sbs_cols if col in df_sbs.columns]]
                df_sbs_output.to_excel(writer, sheet_name="SBS", index=False)
                
                # Apply yellow highlighting
                ws = writer.sheets["SBS"]
                for row in range(2, len(df_sbs_output) + 2):
                    ws.cell(row=row, column=3).fill = yellow_fill  # SBS Code
                    ws.cell(row=row, column=4).fill = yellow_fill  # Short Description
                    ws.cell(row=row, column=5).fill = yellow_fill  # SBS Code Hyphenated
            
            # Write GTIN sheet
            if not df_gtin.empty:
                gtin_cols = ["Service Code", "Service Description",
                            "GTIN Code", "GTIN Ingredients", "GTIN Strength",
                            "Confidence", "Reasoning"]
                df_gtin_output = df_gtin[[col for col in gtin_cols if col in df_gtin.columns]]
                df_gtin_output.to_excel(writer, sheet_name="GTIN", index=False)
                
                # Apply yellow highlighting
                ws = writer.sheets["GTIN"]
                for row in range(2, len(df_gtin_output) + 2):
                    ws.cell(row=row, column=3).fill = yellow_fill  # GTIN Code
                    ws.cell(row=row, column=4).fill = yellow_fill  # GTIN Ingredients
                    ws.cell(row=row, column=5).fill = yellow_fill  # GTIN Strength
            
            # Write GMDN sheet
            if not df_gmdn.empty:
                gmdn_cols = ["Service Code", "Service Description",
                            "GMDN Code", "GMDN Name", "GMDN Definition",
                            "Confidence", "Reasoning"]
                df_gmdn_output = df_gmdn[[col for col in gmdn_cols if col in df_gmdn.columns]]
                df_gmdn_output.to_excel(writer, sheet_name="GMDN", index=False)
                
                # Apply yellow highlighting
                ws = writer.sheets["GMDN"]
                for row in range(2, len(df_gmdn_output) + 2):
                    ws.cell(row=row, column=3).fill = yellow_fill  # GMDN Code
                    ws.cell(row=row, column=4).fill = yellow_fill  # GMDN Name
                    ws.cell(row=row, column=5).fill = yellow_fill  # GMDN Definition

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
