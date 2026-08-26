"""
Context builder for StudyLLM RAG pipeline.
Assembles retrieved chunks into a bounded token budget.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from study_llm.storage.vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """A chunk selected for inclusion in a prompt."""
    text: str
    source: SearchResult


@dataclass
class Citation:
    """Citation metadata shown with an answer."""
    num: int
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    relative_path: str = ""
    score: float = 0.0


class ContextBuilder:
    """Assembles retrieved chunks into LLM context within a token budget."""

    def __init__(self, context_budget_tokens: int = 1500):
        self.context_budget_tokens = context_budget_tokens

    def build_context(
        self,
        results: List[SearchResult],
        question: str = ""
    ) -> tuple:
        selected = []
        citations = []
        used_tokens = 0

        for i, r in enumerate(results, start=1):
            chunk_text = r.text or ""
            est_tokens = max(1, len(chunk_text) // 4)

            if used_tokens + est_tokens > self.context_budget_tokens:
                continue

            selected.append(ContextChunk(text=chunk_text, source=r))
            citations.append(Citation(
                num=i,
                filename=r.filename,
                page=r.page,
                section=r.section,
                relative_path=r.relative_path,
                score=r.score,
            ))
            used_tokens += est_tokens

        logger.debug(
            f"Context built: {len(selected)} chunks, ~{used_tokens} tokens"
        )
        return (selected, citations)
