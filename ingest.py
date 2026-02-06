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
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


def get_embedding_function():
    """Get the appropriate embedding function based on config."""
    # Use sentence-transformers directly (PyTorch-based, no Keras/TensorFlow)
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


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

    # ── SBS Documents ──
    df_sbs = pd.read_excel(xls, sheet_name="SBS V3 Tabular List")
    # Drop rows where SBS Code Hyphenated is NaN/null
    df_sbs = df_sbs.dropna(subset=["SBS Code Hyphenated"])

    for _, row in df_sbs.iterrows():
        short = str(row["Short Description"]).strip()
        long_desc = str(row["Long Description"]).strip()

        # Handle NaN values that become 'nan' strings
        if short.lower() == 'nan':
            short = ""
        if long_desc.lower() == 'nan':
            long_desc = ""

        # Build document text with [SBS] prefix
        if short == long_desc or not long_desc:
            text = f"[SBS] {short}"
        else:
            text = f"[SBS] {short}. {long_desc}"

        docs.append(Document(
            page_content=text,
            metadata={
                "code": str(row["SBS Code Hyphenated"]),
                "system": "SBS",
                "description": short,
            }
        ))
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

        # Convert CODE to string (it's a large integer)
        code_value = row["CODE"]
        if pd.notna(code_value):
            code_str = str(int(code_value))
        else:
            code_str = ""

        docs.append(Document(
            page_content=text,
            metadata={
                "code": code_str,
                "system": "GTIN",
                "description": display,
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

        # Convert termCode to string (it's an integer)
        term_code = row["GMDN_termCode"]
        if pd.notna(term_code):
            code_str = str(int(term_code))
        else:
            code_str = ""

        docs.append(Document(
            page_content=text,
            metadata={
                "code": code_str,
                "system": "GMDN",
                "description": term_name,
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

    # Create vector store from documents
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=persist_dir,
    )

    print(f"Indexed {len(docs)} documents into ChromaDB at '{persist_dir}'")

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
    print("Building documents from reference file...")
    docs = build_documents(config.REFERENCE_FILE)

    print(f"\nBuilt {len(docs)} documents. Sample:")
    for d in docs[:3]:
        print(f"  {d.page_content[:80]}... | code={d.metadata['code']}")

    print("\nCreating vector store...")
    vector_store = create_vector_store(docs)

    print("\nDone! Vector store saved to", config.CHROMA_PERSIST_DIR)

    # Run verification queries
    verify_vector_store(vector_store)
