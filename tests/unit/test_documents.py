"""
Tests for document parsing.
"""

from pathlib import Path
from study_llm.documents.parser import (
    TextParser,
    MarkdownParser,
    parse_document,
    ParsedDocument
)


def test_text_parser_can_parse():
    """Test that TextParser can identify .txt files."""
    parser = TextParser()
    assert parser.can_parse(Path("test.txt")) is True
    assert parser.can_parse(Path("test.pdf")) is False


def test_text_parser_parse():
    """Test parsing a text file."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\n\nThis is the second paragraph.")
        temp_path = Path(f.name)

    try:
        parser = TextParser()
        result = parser.parse(temp_path)

        assert isinstance(result, ParsedDocument)
        assert result.content == "This is a test document.\n\nThis is the second paragraph."
        assert result.error is None
        assert len(result.pages) == 2  # Two paragraphs
        assert result.metadata['word_count'] == 10
    finally:
        os.unlink(temp_path)


def test_markdown_parser_can_parse():
    """Test that MarkdownParser can identify .md files."""
    parser = MarkdownParser()
    assert parser.can_parse(Path("test.md")) is True
    assert parser.can_parse(Path("test.txt")) is False
    assert parser.can_parse(Path("test.pdf")) is False


def test_markdown_parser_parse():
    """Test parsing a markdown file."""
    import tempfile
    import os

    content = "# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        parser = MarkdownParser()
        result = parser.parse(temp_path)

        assert isinstance(result, ParsedDocument)
        assert result.content == content
        assert result.error is None
        assert len(result.sections) == 2  # Title + Section
        assert result.sections[0]['heading'] == "Title"
        assert result.sections[1]['heading'] == "Section"
        assert result.metadata['section_count'] == 2
    finally:
        os.unlink(temp_path)


def test_parse_document_no_parser():
    """Test that parse_document handles unsupported file types gracefully."""
    result = parse_document(Path("test.unknown"))

    assert isinstance(result, ParsedDocument)
    assert result.content == ""
    assert result.error is not None
    assert "No parser available" in result.error