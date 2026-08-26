"""
Text chunking for StudyLLM.
Splits documents into manageable chunks with overlap and metadata.
"""

import re
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from study_llm.documents.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text from a document with metadata."""
    content: str
    chunk_id: str
    document_id: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Chunks parsed documents into smaller pieces with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        respect_sections: bool = False
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            respect_sections: Whether to avoid splitting section boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sections = respect_sections

    def chunk_document(self, doc: ParsedDocument) -> List[Chunk]:
        """
        Chunk a parsed document into smaller pieces.

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of Chunk objects
        """
        if not doc.content:
            logger.warning(f"Empty document content for {doc.filename}")
            return []

        # Generate a document ID from filename (hash computed by caller)
        document_id = self._generate_document_id(doc)

        chunks = []

        if self.respect_sections and doc.sections:
            # Chunk by sections if available
            chunks.extend(self._chunk_by_sections(doc, document_id))
        elif doc.pages:
            # Chunk page by page if available
            chunks.extend(self._chunk_by_pages(doc, document_id))
        else:
            # Fall back to simple character-based chunking
            chunks.extend(self._chunk_by_characters(doc, document_id, doc.content, 0))

        return chunks

    def _generate_document_id(self, doc: ParsedDocument) -> str:
        """Generate a document ID from filename (hash computed by caller)."""
        # For now, use filename + size as a stable identifier
        # Full content hash will be computed by the document manager
        return hashlib.md5(doc.filename.encode()).hexdigest()[:16]

    def _chunk_by_sections(self, doc: ParsedDocument, document_id: str) -> List[Chunk]:
        """Chunk document respecting section boundaries."""
        chunks = []
        chunk_idx = 0

        for section_idx, section in enumerate(doc.sections):
            section_text = section.get('content', '')
            section_heading = section.get('heading', '')

            if not section_text.strip():
                continue

            # Chunk within this section
            section_chunks = self._chunk_by_characters(
                doc, document_id, section_text, chunk_idx,
                section=section_heading,
                page=section.get('page')
            )
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)

        # If no chunks were created (e.g., empty sections), fall back
        if not chunks:
            chunks.extend(self._chunk_by_characters(doc, document_id, doc.content, 0))

        return chunks

    def _chunk_by_pages(self, doc: ParsedDocument, document_id: str) -> List[Chunk]:
        """Chunk document page by page."""
        chunks = []
        chunk_idx = 0

        for page_idx, page_text in enumerate(doc.pages, 1):
            if not page_text.strip():
                continue

            page_chunks = self._chunk_by_characters(
                doc, document_id, page_text, chunk_idx, page=page_idx
            )
            chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        # If no chunks were created (e.g., all empty pages), fall back
        if not chunks:
            chunks.extend(self._chunk_by_characters(doc, document_id, doc.content, 0))

        return chunks

    def _chunk_by_characters(
        self,
        doc: ParsedDocument,
        document_id: str,
        text: str,
        start_idx: int,
        section: Optional[str] = None,
        page: Optional[int] = None
    ) -> List[Chunk]:
        """Chunk text by character count with overlap."""
        chunks = []

        if not text.strip():
            return chunks

        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(sentence) > self.chunk_size:
                chunk = self._create_chunk(
                    doc, document_id, current_chunk, start_idx + len(chunks),
                    current_start, current_start + len(current_chunk),
                    section, page
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text)
            else:
                if not current_chunk:
                    current_start = len(current_chunk)
                current_chunk += sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                doc, document_id, current_chunk, start_idx + len(chunks),
                current_start, current_start + len(current_chunk),
                section, page
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunk boundaries."""
        # Simple sentence splitter
        sentence_enders = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_enders, text)

        # Filter out empty sentences
        return [s for s in sentences if s.strip()]

    def _create_chunk(
        self,
        doc: ParsedDocument,
        document_id: str,
        content: str,
        chunk_idx: int,
        start_char: int,
        end_char: int,
        section: Optional[str],
        page: Optional[int]
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        chunk_id = f"{document_id}-{chunk_idx:04d}"

        return Chunk(
            content=content.strip(),
            chunk_id=chunk_id,
            document_id=document_id,
            filename=doc.filename,
            page=page,
            section=section,
            start_char=start_char,
            end_char=end_char,
            metadata={
                'file_extension': doc.file_extension,
                'parser': doc.metadata.get('parser', 'unknown'),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        )


def chunk_parsed_document(
    doc: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[Chunk]:
    """Convenience function to chunk a parsed document."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(doc)