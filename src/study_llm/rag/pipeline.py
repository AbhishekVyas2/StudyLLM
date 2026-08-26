"""
RAG pipeline for StudyLLM.
Retrieval -> optional rerank -> context build -> LLM answer with citations.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from study_llm.rag.retriever import Retriever
from study_llm.rag.reranker import Reranker, NoOpReranker
from study_llm.rag.context_builder import ContextBuilder, Citation
from study_llm.rag.prompt_template import build_rag_prompt

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """A full RAG answer with provenance."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    chunks_used: int = 0


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(self, retriever: Retriever, inference,
                 reranker: Reranker = None):
        self.retriever = retriever
        self.inference = inference
        self.reranker = reranker or NoOpReranker()
        cfg_budget = 1500
        try:
            from study_llm.core.config import get_config
            cfg_budget = get_config().context_budget_tokens
        except Exception:
            pass
        self.context_builder = ContextBuilder(context_budget_tokens=cfg_budget)

    def ask(self, question: str) -> RAGAnswer:
        """Answer a question using retrieved document context."""
        results = self.retriever.retrieve(question)

        if results:
            results = self.reranker.rerank(question, results)

        chunks, citations = self.context_builder.build_context(results, question)

        if not chunks:
            return RAGAnswer(
                question=question,
                answer="I don't know based on your documents.",
                citations=[],
                chunks_used=0,
            )

        context_text = "\n\n".join(
            f"[{c.num}] {c.filename}" + (f" (p.{c.page})" if c.page else "")
            + f"\n{chunk.text}"
            for c, chunk in zip(citations, chunks)
        )
        prompt = build_rag_prompt(context_text, question)
        answer = self.inference.generate(prompt, max_tokens=512)

        return RAGAnswer(
            question=question,
            answer=answer.strip(),
            citations=citations,
            chunks_used=len(chunks),
        )
