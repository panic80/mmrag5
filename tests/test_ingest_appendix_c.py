import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import pytest

from ingest_rag import load_documents

@pytest.mark.integration
def test_extract_appendix_c_tables():
    """Test that load_documents extracts and chunks Appendix C tables from the NJC directive."""
    url = "https://www.njc-cnm.gc.ca/directive/d10/v238/en?print"
    # Use a large chunk_size to avoid splitting table across chunks
    docs = load_documents(url, chunk_size=20000, overlap=0)
    # Find chunks containing the Appendix C header
    table_chunks = [doc for doc in docs if "Appendix C - Allowances" in doc.content]
    assert table_chunks, "No chunk with 'Appendix C - Allowances' header found"
    # Combine content for inspection
    content = "\n".join(doc.content for doc in table_chunks)
    # Assert that the table header appears
    assert "Canada & USA" in content, "Table header 'Canada & USA' not found in extracted content"
    # Assert that multiple numeric entries are present
    numeric_matches = re.findall(r"\b\d+\.\d+\b", content)
    assert len(numeric_matches) > 10, f"Expected multiple numeric entries, found {len(numeric_matches)}"