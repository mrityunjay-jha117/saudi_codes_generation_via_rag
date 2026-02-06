"""
app.py - Streamlit UI for Saudi Billing Code Matcher.

A minimal single-page UI with:
1. Sidebar: Reference data upload and indexing
2. Main area: Code mapping sheet processing
3. Single test match input
"""

# Set environment variable BEFORE any TensorFlow imports to use legacy Keras
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import os
import tempfile
import json
import config

# Page configuration
st.set_page_config(
    page_title="Saudi Code Matcher",
    page_icon="",
    layout="wide",
)

# Ensure output directory exists
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


@st.cache_resource(ttl=60)  # Cache for 60 seconds only, then reload
def get_matcher():
    """Get or create CodeMatcher instance (short-lived cache)."""
    from matcher import CodeMatcher
    return CodeMatcher()


def check_vector_db_exists() -> bool:
    """Check if the vector database has been populated."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        collection = client.get_collection(config.COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


# ============== SIDEBAR: Reference Data ==============
with st.sidebar:
    st.header("Reference Data")

    # Show vector DB status
    if check_vector_db_exists():
        st.success("Vector DB Ready")
    else:
        st.warning("No Vector DB - upload reference data first")

    st.divider()

    # Reference file uploader
    ref_file = st.file_uploader(
        "Upload Saudi Billing Codes Excel",
        type=["xlsx"],
        help="The reference file containing SBS, GTIN, and GMDN codes"
    )

    if ref_file and st.button("Index Reference Data", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(ref_file.read())
            ref_path = f.name

        with st.spinner("Building vector index..."):
            try:
                from ingest import build_documents, create_vector_store

                docs = build_documents(ref_path)
                create_vector_store(docs)

                st.success(f"Indexed {len(docs)} codes!")

                # Clear the cached matcher so it reloads
                get_matcher.clear()

            except Exception as e:
                st.error(f"Error indexing: {str(e)}")
            finally:
                # Clean up temp file (with error handling)
                try:
                    os.unlink(ref_path)
                except Exception:
                    pass  # Ignore if file is still in use

    st.divider()

    # Configuration info
    st.subheader("Configuration")
    if config.USE_LOCAL:
        st.info("Using LOCAL models (no Gemini API key found)")
        st.caption("Embedding: all-MiniLM-L6-v2 (HuggingFace)")
        st.caption("LLM: Ollama gemma3:1b")
    else:
        st.info("Using Google Gemini API")
        st.caption("Embedding: all-MiniLM-L6-v2 (HuggingFace - Free)")
        st.caption(f"LLM: {config.LLM_MODEL}")

    st.caption(f"Top-K retrieval: {config.TOP_K}")
    st.caption(f"Auto-accept threshold: {config.AUTO_ACCEPT_THRESHOLD}")


# ============== MAIN AREA ==============
st.title("Saudi Billing Code Matcher")
st.caption("RAG-powered matching of service descriptions to standardized billing codes")

# ============== Code Matching Section ==============
st.header("Code Matching")

mapping_file = st.file_uploader(
    "Upload Code Mapping Sheet",
    type=["xlsx"],
    help="Excel file with Service Code and Service Description columns"
)

col1, col2 = st.columns([1, 3])
with col1:
    use_async = st.checkbox("Use async processing", value=False,
                            help="Faster but requires more resources")

if mapping_file and st.button("Match Codes", type="primary", use_container_width=True):
    if not check_vector_db_exists():
        st.error("Please index reference data first (see sidebar)")
    else:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(mapping_file.read())
            input_path = f.name

        # Generate output filename
        output_filename = f"matched_{mapping_file.name}"
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)

        # Progress display
        progress_container = st.empty()
        status_text = st.empty()

        with st.spinner("Matching codes... This may take a while for large files."):
            try:
                matcher = get_matcher()

                if use_async:
                    import asyncio
                    results = asyncio.run(matcher.match_batch_async(input_path, output_path))
                else:
                    results = matcher.match_batch(input_path, output_path)

                st.success("Matching complete!")

                # Show summary stats
                st.subheader("Summary")
                for sheet_name, df in results:
                    with st.expander(f"Sheet: {sheet_name} ({len(df)} rows)", expanded=True):
                        # Confidence distribution
                        conf_counts = df["Confidence"].value_counts()
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("HIGH", conf_counts.get("HIGH", 0))
                        col2.metric("MEDIUM", conf_counts.get("MEDIUM", 0))
                        col3.metric("LOW", conf_counts.get("LOW", 0))
                        col4.metric("NONE", conf_counts.get("NONE", 0))

                        # Preview data
                        st.dataframe(df.head(10), use_container_width=True)

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Matched Results",
                        data=f,
                        file_name=output_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

            except Exception as e:
                st.error(f"Error during matching: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # Clean up temp file
        os.unlink(input_path)

# ============== Single Test Match Section ==============
st.divider()
st.header("Test Single Description")
st.caption("Quickly test matching for individual service descriptions")

test_input = st.text_input(
    "Enter a service description",
    placeholder="e.g., Cisternal puncture, BEPANTHEN, vitamin D3 IVD calibrator"
)

if test_input and st.button("Match", key="single_match"):
    if not check_vector_db_exists():
        st.error("Please index reference data first (see sidebar)")
    else:
        with st.spinner("Matching..."):
            try:
                matcher = get_matcher()
                result = matcher.match_single(test_input)

                # Display result
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Match Result")
                    if result["matched_code"]:
                        st.metric("Matched Code", result["matched_code"])
                        st.metric("Code System", result["code_system"])
                        st.metric("Confidence", result["confidence"])
                    else:
                        st.warning("No match found")

                with col2:
                    st.subheader("Details")
                    if result["matched_description"]:
                        st.write(f"**Description:** {result['matched_description']}")
                    st.write(f"**Reasoning:** {result.get('reasoning', 'N/A')}")

                # Show raw JSON
                with st.expander("Raw JSON Response"):
                    st.json(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")


# ============== Footer ==============
st.divider()
st.caption("Saudi Billing Code Matcher - RAG POC | Built with LangChain + ChromaDB + Streamlit")
