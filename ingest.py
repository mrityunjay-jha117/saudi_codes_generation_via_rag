"""
ingest.py - Read reference Excel and build ChromaDB vector store.

This file does three things:
1. Reads the reference Excel file (SBS, GTIN, GMDN sheets)
2. Builds LangChain Document objects with proper text/metadata
3. Embeds them into ChromaDB
"""

# Set environment variable BEFORE any TensorFlow imports to use legacy Keras
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
import config


class SentenceTransformerEmbeddings:
    """Custom embedding class using sentence-transformers (PyTorch) for LangChain compatibility."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents with batch processing."""
        # Use batch processing with progress bar for better performance
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=32,  # Optimal batch size for sentence-transformers
            show_progress_bar=True  # Show progress during embedding
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


def get_embedding_function():
    """Get the appropriate embedding function based on config."""
    # Use sentence-transformers directly (PyTorch-based, no Keras/TensorFlow)
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def clean_field(value) -> str:
    """Clean a DataFrame cell value, returning empty string for null/nan."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def build_enriched_sbs_document(row: pd.Series) -> Document | None:
    """
    Build a richly annotated Document from an SBS reference row.
    Uses ALL available columns: Specialty (Chapter), Category (Block),
    Clinical Explanation, Includes, Excludes, Guideline.
    Returns None for empty/invalid rows.
    """
    code = clean_field(row.get("SBS Code Hyphenated"))
    short = clean_field(row.get("Short Description"))

    if not code or not short:
        return None

    long_desc = clean_field(row.get("Long Description"))
    chapter = clean_field(row.get("Chapter Name"))
    block = clean_field(row.get("Block Name"))
    clinical = clean_field(row.get("Cliincal Explanation "))  # Note: typo + trailing space in source data
    includes = clean_field(row.get("Includes"))
    excludes = clean_field(row.get("Excludes"))
    guideline = clean_field(row.get("Guideline"))
    chapter_num = row.get("Chapter number")
    block_num = clean_field(row.get("Block Number"))

    # Build enriched page_content
    parts = [f"[SBS] Code: {code}"]
    if chapter:
        parts.append(f"Specialty: {chapter}")
    if block:
        parts.append(f"Category: {block}")
    parts.append(f"Description: {short}")
    if long_desc and long_desc != short:
        parts.append(f"Detail: {long_desc}")
    if clinical:
        parts.append(f"Clinical context: {clinical}")
    if includes:
        parts.append(f"Includes: {includes}")
    if excludes:
        parts.append(f"Excludes: {excludes}")
    if guideline:
        parts.append(f"Guideline: {guideline}")

    page_content = ". ".join(parts)

    # Get SBS Code (numeric) if available
    sbs_code_numeric = ""
    if "SBS Code" in row.index and pd.notna(row.get("SBS Code")):
        try:
            sbs_code_numeric = str(int(float(row["SBS Code"])))
        except (ValueError, TypeError):
            sbs_code_numeric = str(row["SBS Code"]).strip()

    # Build metadata for filtered retrieval
    metadata = {
        "code": code,
        "code_hyphenated": code,
        "code_numeric": sbs_code_numeric,
        "system": "SBS",
        "description": short,
        "chapter_name": chapter.upper().strip() if chapter else "",
        "chapter_number": float(chapter_num) if pd.notna(chapter_num) else -1,
        "block_name": block,
        "block_number": block_num,
        "has_excludes": bool(excludes),
        "has_includes": bool(includes),
        "has_clinical_explanation": bool(clinical),
    }

    return Document(page_content=page_content, metadata=metadata)


def build_documents(excel_path: str = None) -> list[Document]:
    """
    Read reference Excel and build LangChain Documents.

    Args:
        excel_path: Path to the reference Excel file. Defaults to config.REFERENCE_FILE.

    Returns:
        List of LangChain Document objects ready for embedding.
    """
    if excel_path is None:
        excel_path = config.REFERENCE_FILE

    xls = pd.ExcelFile(excel_path)
    docs = []

    # Track counts per system
    counts = {"SBS": 0, "GTIN": 0, "GMDN": 0}

    # ── SBS Documents (ENRICHED with all metadata) ──
    df_sbs = pd.read_excel(xls, sheet_name="SBS V3 Tabular List")
    df_sbs = df_sbs.dropna(subset=["SBS Code Hyphenated"])

    for _, row in df_sbs.iterrows():
        doc = build_enriched_sbs_document(row)
        if doc:
            docs.append(doc)
            counts["SBS"] += 1

    # ── GTIN Documents ──
    df_gtin = pd.read_excel(xls, sheet_name="GTIN")

    for _, row in df_gtin.iterrows():
        display = str(row["DISPLAY"]).strip()
        ingredients = str(row["INGREDIENTS"]).strip()
        strength = str(row["STRENGTH"]).strip()

        # Handle NaN values
        if display.lower() == 'nan':
            display = ""
        if ingredients.lower() == 'nan':
            ingredients = ""
        if strength.lower() == 'nan':
            strength = ""

        # Build document text with [GTIN] prefix
        text = f"[GTIN] {display} - {ingredients} {strength}"

        # Convert CODE to string (handle both numeric and text codes)
        code_value = row["CODE"]
        if pd.notna(code_value):
            try:
                # Try to convert to int first (for numeric GTIN codes)
                code_str = str(int(float(code_value)))
            except (ValueError, TypeError):
                # If it's already a string or can't be converted, use as-is
                code_str = str(code_value).strip()
        else:
            code_str = ""

        docs.append(Document(
            page_content=text,
            metadata={
                "code": code_str,
                "system": "GTIN",
                "description": display,
                "ingredients": ingredients,
                "strength": strength,
            }
        ))
        counts["GTIN"] += 1

    # ── GMDN Documents ──
    df_gmdn = pd.read_excel(xls, sheet_name="GMDN")

    for _, row in df_gmdn.iterrows():
        term_name = str(row["termName"]).strip()
        term_def = str(row["termDefinition"]).strip()

        # Handle NaN values
        if term_name.lower() == 'nan':
            term_name = ""
        if term_def.lower() == 'nan':
            term_def = ""

        # Truncate termDefinition to 300 chars max
        term_def_short = term_def[:300] if len(term_def) > 300 else term_def

        # Build document text with [GMDN] prefix
        text = f"[GMDN] {term_name}. {term_def_short}"

        # Convert termCode to string (handle both numeric and text codes)
        term_code = row["GMDN_termCode"]
        if pd.notna(term_code):
            try:
                # Try to convert to int first (for numeric GMDN codes)
                code_str = str(int(float(term_code)))
            except (ValueError, TypeError):
                # If it's already a string or can't be converted, use as-is
                code_str = str(term_code).strip()
        else:
            code_str = ""

        docs.append(Document(
            page_content=text,
            metadata={
                "code": code_str,
                "system": "GMDN",
                "description": term_name,
                "term_name": term_name,
                "term_definition": term_def,
            }
        ))
        counts["GMDN"] += 1

    print(f"SBS: {counts['SBS']}, GTIN: {counts['GTIN']}, GMDN: {counts['GMDN']}, Total: {len(docs)}")

    return docs


def create_vector_store(docs: list[Document], persist_dir: str = None) -> Chroma:
    """
    Embed documents and persist to ChromaDB.

    Args:
        docs: List of LangChain Document objects to embed.
        persist_dir: Directory to persist ChromaDB. Defaults to config.CHROMA_PERSIST_DIR.

    Returns:
        The Chroma vector store instance.
    """
    if persist_dir is None:
        persist_dir = config.CHROMA_PERSIST_DIR

    # Delete existing collection to avoid duplicates on re-run
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(config.COLLECTION_NAME)
        print(f"Deleted existing collection '{config.COLLECTION_NAME}'")
    except Exception:
        pass  # Collection doesn't exist yet

    # Get embedding function based on config
    embeddings = get_embedding_function()

    print(f"\nIndexing {len(docs)} documents into ChromaDB...")
    print("Using batch processing for optimal performance...")
    
    # Create vector store with batch processing for better performance
    # Process in batches to avoid memory issues and improve speed
    batch_size = 500  # Optimal batch size for ChromaDB
    
    vector_store = None
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        if vector_store is None:
            # Create new vector store with first batch
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=config.COLLECTION_NAME,
                persist_directory=persist_dir,
            )
        else:
            # Add subsequent batches to existing vector store
            vector_store.add_documents(batch)
    
    print(f"Successfully indexed {len(docs)} documents into ChromaDB at '{persist_dir}'")

    return vector_store


def verify_vector_store(vector_store: Chroma = None):
    """
    Verify the vector store by running test queries.

    Args:
        vector_store: The Chroma vector store to test. If None, loads from disk.
    """
    if vector_store is None:
        embeddings = get_embedding_function()
        vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

    print("\n--- Verification Queries ---")

    # Test query 1: "puncture" should return SBS results
    print("\nQuery: 'puncture' (expecting SBS results)")
    results = vector_store.similarity_search("puncture", k=3)
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['system']}] {doc.metadata['code']} | {doc.page_content[:60]}...")

    # Test query 2: "BEPANTHEN" should return GTIN results
    print("\nQuery: 'BEPANTHEN' (expecting GTIN results)")
    results = vector_store.similarity_search("BEPANTHEN", k=3)
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['system']}] {doc.metadata['code']} | {doc.page_content[:60]}...")

    # Test query 3: "vitamin D3" should return GMDN results
    print("\nQuery: 'vitamin D3' (expecting GMDN results)")
    results = vector_store.similarity_search("vitamin D3", k=3)
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['system']}] {doc.metadata['code']} | {doc.page_content[:60]}...")


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    print("=" * 60)
    print("SAUDI BILLING CODE INDEXING")
    print("=" * 60)
    
    print("\nBuilding documents from reference file...")
    docs = build_documents(config.REFERENCE_FILE)

    print(f"\nBuilt {len(docs)} documents. Sample:")
    for d in docs[:3]:
        print(f"  {d.page_content[:80]}... | code={d.metadata['code']}")

    print("\nCreating vector store...")
    vector_store = create_vector_store(docs)

    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Vector store saved to: {config.CHROMA_PERSIST_DIR}")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Speed: {len(docs)/elapsed:.1f} documents/second")
    print("=" * 60)

    # Run verification queries
    print("\nRunning verification queries...")
    verify_vector_store(vector_store)
