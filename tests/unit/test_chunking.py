"""
Tests for chunking.
"""

from pathlib import Path
from study_llm.documents.parser import ParsedDocument
from study_llm.chunking.text_chunker import TextChunker, Chunk


def create_test_document(content: str, pages=None, sections=None) -> ParsedDocument:
    """Create a test document with given content."""
    return ParsedDocument(
        source_path=Path("test.txt"),
        filename="test.txt",
        file_extension=".txt",
        content=content,
        pages=pages,
        sections=sections,
        metadata={'word_count': len(content.split())}
    )


def test_chunker_basic():
    """Test basic chunking functionality."""
    doc = create_test_document("This is a test. " * 100)  # ~1500 chars

    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.content for c in chunks)
    assert all(c.document_id for c in chunks)
    assert all(c.chunk_id for c in chunks)


def test_chunker_respects_size():
    """Test that chunks respect target size (approximately)."""
    doc = create_test_document("Word " * 1000)  # 5000 chars

    chunk_size = 512
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    # Allow some flexibility for sentence boundaries
    for chunk in chunks[:-1]:  # Last chunk can be smaller
        assert len(chunk.content) <= chunk_size + 100  # Allow 100 char flexibility


def test_chunker_preserves_overlap():
    """Test that chunks have overlap."""
    doc = create_test_document(
        " ".join(f"Sentence number {i} is here." for i in range(100))
    )

    chunk_size = 200
    overlap = 50
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.chunk_document(doc)

    # Check that there's some overlap between consecutive chunks
    # (content from end of one chunk appears in start of next)
    if len(chunks) >= 2:
        # Get last 'overlap' chars of first chunk
        first_end = chunks[0].content[-overlap:]
        # Check if it appears in second chunk
        assert first_end in chunks[1].content or overlap == 0


def test_chunker_with_pages():
    """Test chunking with page information."""
    pages = ["Page 1 content here. " * 20, "Page 2 content here. " * 20]
    doc = create_test_document(" ".join(pages), pages=pages)

    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    # Check that pages are assigned
    pages_assigned = {c.page for c in chunks if c.page is not None}
    assert len(pages_assigned) > 0


def test_chunker_with_sections():
    """Test chunking with section information."""
    sections = [
        {'heading': 'Introduction', 'content': 'Intro text here. ' * 30},
        {'heading': 'Methods', 'content': 'Methods text here. ' * 30}
    ]
    doc = create_test_document("Full content. " * 100, sections=sections)

    chunker = TextChunker(chunk_size=512, chunk_overlap=50, respect_sections=True)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    # Check that sections are assigned
    sections_assigned = {c.section for c in chunks if c.section is not None}
    assert 'Introduction' in sections_assigned
    assert 'Methods' in sections_assigned


def test_chunker_empty_document():
    """Test chunking empty document."""
    doc = create_test_document("")

    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 0


def test_chunker_metadata():
    """Test that chunks contain correct metadata."""
    doc = create_test_document("Test content for metadata. " * 50)

    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    chunk = chunks[0]

    assert chunk.filename == "test.txt"
    assert chunk.metadata['file_extension'] == ".txt"
    assert chunk.metadata['chunk_size'] == 512
    assert chunk.metadata['chunk_overlap'] == 50
    assert chunk.start_char >= 0
    assert chunk.end_char > chunk.start_char