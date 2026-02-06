"""
test_matcher.py - Tests for the Saudi Billing Code Matcher.

Run with: pytest test_matcher.py -v
"""

import pytest
from langchain_core.documents import Document
import config


class TestBuildDocuments:
    """Tests for the document building functionality."""

    def test_build_documents_count(self):
        """Test that the correct number of documents are built."""
        from ingest import build_documents

        docs = build_documents(config.REFERENCE_FILE)

        # Expected: 9 SBS + 9 GTIN + 8 GMDN = 26 documents
        assert len(docs) == 26, f"Expected 26 documents, got {len(docs)}"

    def test_build_documents_have_required_metadata(self):
        """Test that all documents have required metadata fields."""
        from ingest import build_documents

        docs = build_documents(config.REFERENCE_FILE)

        for doc in docs:
            assert "code" in doc.metadata, "Document missing 'code' metadata"
            assert "system" in doc.metadata, "Document missing 'system' metadata"
            assert doc.metadata["code"], "Document has empty 'code' metadata"
            assert doc.metadata["system"], "Document has empty 'system' metadata"

    def test_build_documents_no_nan_content(self):
        """Test that no document has NaN in page_content."""
        from ingest import build_documents

        docs = build_documents(config.REFERENCE_FILE)

        for doc in docs:
            assert "nan" not in doc.page_content.lower() or "[" in doc.page_content, \
                f"Document contains NaN: {doc.page_content[:50]}"

    def test_build_documents_have_system_prefix(self):
        """Test that documents have correct system prefixes."""
        from ingest import build_documents

        docs = build_documents(config.REFERENCE_FILE)

        sbs_docs = [d for d in docs if d.metadata["system"] == "SBS"]
        gtin_docs = [d for d in docs if d.metadata["system"] == "GTIN"]
        gmdn_docs = [d for d in docs if d.metadata["system"] == "GMDN"]

        for doc in sbs_docs:
            assert doc.page_content.startswith("[SBS]"), \
                f"SBS doc missing prefix: {doc.page_content[:30]}"

        for doc in gtin_docs:
            assert doc.page_content.startswith("[GTIN]"), \
                f"GTIN doc missing prefix: {doc.page_content[:30]}"

        for doc in gmdn_docs:
            assert doc.page_content.startswith("[GMDN]"), \
                f"GMDN doc missing prefix: {doc.page_content[:30]}"


class TestDocumentMetadataTypes:
    """Tests for metadata type correctness."""

    def test_all_codes_are_strings(self):
        """Test that all metadata code values are strings."""
        from ingest import build_documents

        docs = build_documents(config.REFERENCE_FILE)

        for doc in docs:
            code = doc.metadata["code"]
            assert isinstance(code, str), \
                f"Code is not a string: {type(code)} - {code}"

    def test_all_systems_are_valid(self):
        """Test that all system values are one of SBS, GTIN, or GMDN."""
        from ingest import build_documents

        docs = build_documents(config.REFERENCE_FILE)
        valid_systems = {"SBS", "GTIN", "GMDN"}

        for doc in docs:
            system = doc.metadata["system"]
            assert system in valid_systems, \
                f"Invalid system: {system}"


class TestFormatCandidates:
    """Tests for the candidate formatting function."""

    def test_format_candidates_output(self):
        """Test that format_candidates produces correctly formatted output."""
        from prompts import format_candidates

        # Create mock documents
        mock_docs = [
            Document(
                page_content="[SBS] Cisternal puncture",
                metadata={"code": "39003-00-00", "system": "SBS", "description": "Cisternal puncture"}
            ),
            Document(
                page_content="[GTIN] BEPANTHEN - DEXPANTHENOL 5 G/ 100 G",
                metadata={"code": "6285074000864", "system": "GTIN", "description": "BEPANTHEN"}
            ),
        ]

        result = format_candidates(mock_docs)

        # Check it's a numbered list
        lines = result.strip().split("\n")
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        # Check format
        assert lines[0].startswith("1."), "First line should start with '1.'"
        assert "[SBS]" in lines[0], "First line should contain [SBS]"
        assert "39003-00-00" in lines[0], "First line should contain the code"

        assert lines[1].startswith("2."), "Second line should start with '2.'"
        assert "[GTIN]" in lines[1], "Second line should contain [GTIN]"
        assert "6285074000864" in lines[1], "Second line should contain the code"


@pytest.mark.integration
class TestVectorStoreRetrieval:
    """Integration tests that require a populated ChromaDB."""

    def test_puncture_query_returns_sbs(self):
        """Test that 'puncture' query returns SBS results."""
        from ingest import get_embedding_function
        from langchain_community.vectorstores import Chroma

        embeddings = get_embedding_function()
        vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        results = vector_store.similarity_search("puncture", k=3)
        assert len(results) > 0, "No results returned for 'puncture'"

        top_result = results[0]
        assert top_result.metadata["system"] == "SBS", \
            f"Expected SBS system, got {top_result.metadata['system']}"
        assert "39" in top_result.metadata["code"], \
            f"Expected code containing '39', got {top_result.metadata['code']}"

    def test_bepanthen_query_returns_gtin(self):
        """Test that 'BEPANTHEN' query returns GTIN results."""
        from ingest import get_embedding_function
        from langchain_community.vectorstores import Chroma

        embeddings = get_embedding_function()
        vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        results = vector_store.similarity_search("BEPANTHEN", k=3)
        assert len(results) > 0, "No results returned for 'BEPANTHEN'"

        top_result = results[0]
        assert top_result.metadata["system"] == "GTIN", \
            f"Expected GTIN system, got {top_result.metadata['system']}"
        assert top_result.metadata["code"].startswith("628"), \
            f"Expected code starting with '628', got {top_result.metadata['code']}"

    def test_vitamin_d3_query_returns_gmdn(self):
        """Test that 'vitamin D3' query returns GMDN results."""
        from ingest import get_embedding_function
        from langchain_community.vectorstores import Chroma

        embeddings = get_embedding_function()
        vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        results = vector_store.similarity_search("vitamin D3", k=3)
        assert len(results) > 0, "No results returned for 'vitamin D3'"

        top_result = results[0]
        assert top_result.metadata["system"] == "GMDN", \
            f"Expected GMDN system, got {top_result.metadata['system']}"
        assert top_result.metadata["code"] in ["38242", "38243"], \
            f"Expected code 38242 or 38243, got {top_result.metadata['code']}"


class TestMatchSingleFallback:
    """Tests for the JSON parse fallback behavior."""

    def test_match_single_handles_empty_description(self):
        """Test that match_single handles empty descriptions gracefully."""
        from matcher import CodeMatcher

        matcher = CodeMatcher()

        result = matcher.match_single("")
        assert result["confidence"] == "NONE"
        assert result["matched_code"] is None

        result = matcher.match_single(None)
        assert result["confidence"] == "NONE"
        assert result["matched_code"] is None

    def test_parse_llm_response_fallback(self):
        """Test the JSON parse fallback mechanism."""
        from matcher import CodeMatcher
        from langchain_core.documents import Document

        matcher = CodeMatcher()

        # Create a mock fallback document
        fallback_doc = Document(
            page_content="[SBS] Test procedure",
            metadata={"code": "12345-00-00", "system": "SBS", "description": "Test procedure"}
        )

        # Test with invalid JSON
        result = matcher._parse_llm_response("not valid json", fallback_doc)
        assert result["confidence"] == "LOW"
        assert result["matched_code"] == "12345-00-00"
        assert "parsing failed" in result["reasoning"].lower()

        # Test with markdown-wrapped JSON
        valid_json = '{"matched_code": "99999-00-00", "code_system": "SBS", "matched_description": "Test", "confidence": "HIGH", "reasoning": "Test"}'
        markdown_wrapped = f"```json\n{valid_json}\n```"

        result = matcher._parse_llm_response(markdown_wrapped, fallback_doc)
        assert result["matched_code"] == "99999-00-00"
        assert result["confidence"] == "HIGH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
