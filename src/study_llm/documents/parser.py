"""
Document parsing abstraction for StudyLLM.
Provides a common interface for parsing various document formats.
"""

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ParsedDocument:
    """Normalized representation of a parsed document."""
    source_path: Path
    filename: str
    file_extension: str
    content: str
    pages: Optional[List[str]] = None  # Page-by-page text if available
    sections: Optional[List[Dict[str, Any]]] = None  # Structured sections
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class DocumentParser(abc.ABC):
    """Abstract base class for document parsers."""

    @abc.abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Return True if this parser can handle the given file."""
        pass

    @abc.abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse the file and return a normalized document representation."""
        pass


class TextParser(DocumentParser):
    """Parser for plain text files."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.txt', '.text']

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            return ParsedDocument(
                source_path=file_path,
                filename=file_path.name,
                file_extension=file_path.suffix.lower(),
                content=content,
                pages=content.split('\n\n'),  # Split into paragraphs as pseudo-pages
                metadata={
                    'parser': 'text',
                    'word_count': len(content.split())
                }
            )
        except Exception as e:
            logger.error(f"Failed to parse text file {file_path}: {e}")
            return ParsedDocument(
                source_path=file_path,
                filename=file_path.name,
                file_extension=file_path.suffix.lower(),
                content="",
                error=str(e)
            )


class MarkdownParser(DocumentParser):
    """Parser for Markdown files."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.md', '.markdown']

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Simple section extraction based on headers
            sections = []
            current_section = {'heading': 'Introduction', 'content': ''}
            for line in content.split('\n'):
                if line.startswith('#'):
                    if current_section['content']:
                        sections.append(current_section)
                    current_section = {
                        'heading': line.lstrip('#').strip(),
                        'content': ''
                    }
                else:
                    current_section['content'] += line + '\n'

            if current_section['content']:
                sections.append(current_section)

            return ParsedDocument(
                source_path=file_path,
                filename=file_path.name,
                file_extension=file_path.suffix.lower(),
                content=content,
                sections=sections,
                metadata={
                    'parser': 'markdown',
                    'word_count': len(content.split()),
                    'section_count': len(sections)
                }
            )
        except Exception as e:
            logger.error(f"Failed to parse markdown file {file_path}: {e}")
            return ParsedDocument(
                source_path=file_path,
                filename=file_path.name,
                file_extension=file_path.suffix.lower(),
                content="",
                error=str(e)
            )


class DoclingParser(DocumentParser):
    """Parser using Docling for advanced document processing."""

    def __init__(self):
        self._docling_available = self._check_docling()

    def _check_docling(self) -> bool:
        """Check if docling is available."""
        try:
            import docling
            return True
        except ImportError:
            logger.warning("docling is not installed. "
                          "Install it with: pip install docling")
            return False

    def can_parse(self, file_path: Path) -> bool:
        if not self._docling_available:
            return False
        return file_path.suffix.lower() in [
            '.pdf', '.docx', '.pptx', '.xlsx', '.html', '.csv'
        ]

    def parse(self, file_path: Path) -> ParsedDocument:
        from study_llm.documents.docling_adapter import parse_with_docling
        return parse_with_docling(file_path)


class ParserFactory:
    """Factory for creating appropriate document parsers."""

    _parsers = None

    @classmethod
    def get_parsers(cls) -> List[DocumentParser]:
        """Get all available parsers."""
        if cls._parsers is None:
            cls._parsers = [
                TextParser(),
                MarkdownParser(),
                DoclingParser(),
            ]
        return cls._parsers

    @classmethod
    def get_parser_for_file(cls, file_path: Path) -> Optional[DocumentParser]:
        """Get the appropriate parser for a file."""
        for parser in cls.get_parsers():
            if parser.can_parse(file_path):
                return parser
        return None


logger = logging.getLogger(__name__)


def parse_document(file_path: Path) -> ParsedDocument:
    """Parse a document using the appropriate parser."""
    parser = ParserFactory.get_parser_for_file(file_path)
    if parser is None:
        return ParsedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_extension=file_path.suffix.lower(),
            content="",
            error="No parser available for this file type"
        )
    return parser.parse(file_path)
