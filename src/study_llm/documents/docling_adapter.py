"""
Docling adapter: real Docling API -> StudyLLM ParsedDocument.

Uses DocumentConverter (public entry point since docling 2.x).
Text is assembled from the flat item list; each item's provenance
carries its page number. Section headings come from items labelled
section_header / title.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DOCLING_AVAILABLE = None


def _docling_available() -> bool:
    global _DOCLING_AVAILABLE
    if _DOCLING_AVAILABLE is None:
        try:
            import docling  # noqa: F401
            _DOCLING_AVAILABLE = True
        except ImportError:
            logger.warning(
                "docling is not installed. Install it with: pip install docling"
            )
            _DOCLING_AVAILABLE = False
    return _DOCLING_AVAILABLE


def parse_with_docling(file_path: Path):
    """Parse file_path with Docling.

    Returns a ParsedDocument with content/pages/sections filled,
    or one with .error set on failure.
    """
    from study_llm.documents.parser import ParsedDocument

    if not _docling_available():
        return ParsedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_extension=file_path.suffix.lower(),
            content="",
            error="docling not available",
        )

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document

        # Assemble text from the flat item list, tracking page numbers.
        # doc.pages is a dict keyed by 1-based page number.
        page_numbers = sorted(doc.pages.keys())
        pages: list = [""] * (max(page_numbers) if page_numbers else 0)
        sections: list = []

        for item, _level in doc.iterate_items():
            label = getattr(item, "label", None)
            label_name = getattr(label, "value", None) or (
                getattr(label, "name", "") if label else ""
            )
            # Some docling versions don't populate .label on heading items;
            # the class name carries the same information.
            type_name = type(item).__name__.lower()
            is_heading = label_name in ("section_header", "title") or                 "sectionheader" in type_name or "title" in type_name
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue

            provs = getattr(item, "prov", []) or []
            page_no = provs[0].page_no if provs else (
                page_numbers[0] if page_numbers else 1
            )
            idx = page_no - 1 if 1 <= page_no <= len(pages) else None
            if idx is not None:
                pages[idx] += ("\n" if pages[idx] else "") + text

            if is_heading:
                sections.append({
                    "heading": text,
                    "content": "",
                    "page": page_no,
                })

        full_text = "\n\n".join(p for p in pages if p.strip())

        return ParsedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_extension=file_path.suffix.lower(),
            content=full_text or "\n".join(
                getattr(it, "text", "") for it, _ in []
            ),
            pages=pages or [full_text],
            sections=sections,
            metadata={
                "parser": "docling",
                "page_count": max(len(pages), 1),
                "word_count": len(full_text.split()),
            },
        )
    except Exception as e:
        logger.error(f"Failed to parse with docling {file_path}: {e}")
        return ParsedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_extension=file_path.suffix.lower(),
            content="",
            error=str(e),
        )
