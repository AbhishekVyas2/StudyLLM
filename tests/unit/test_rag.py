"""
Tests for the RAG pipeline (retriever, reranker, context builder, pipeline).
"""

from unittest.mock import Mock
import pytest

from study_llm.storage.vector_store import SearchResult
from study_llm.rag.context_builder import ContextBuilder, Citation


def make_result(chunk_id="c1", doc_id="d1", text="Paris is the capital of France.",
                score=0.9):
    return SearchResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        score=score,
        filename="world.txt",
        page=1,
        section=None,
        text=text,
    )


def test_context_builder_respects_budget():
    results = [make_result(text="x" * 4000)]  # ~1000 tokens each
    cb = ContextBuilder(context_budget_tokens=1500)
    chunks, citations = cb.build_context(results)
    assert len(chunks) == 1


def test_context_builder_skips_oversized_chunk():
    results = [make_result(text="x" * 40000)]  # way over budget
    cb = ContextBuilder(context_budget_tokens=1500)
    chunks, citations = cb.build_context(results)
    assert len(chunks) == 0


def test_context_builder_citation_fields():
    r = make_result()
    cb = ContextBuilder(context_budget_tokens=1500)

    chunks, citations = cb.build_context([r])
    assert len(chunks) == 1 and len(citations) == 1
    assert citations[0].num == 1
    assert citations[0].filename == "world.txt"
    assert citations[0].score > 0.5


def test_context_builder_multiple_chunks_numbered():
    rs = [make_result(chunk_id=f"c{i}", score=0.9 - i * 0.1) for i in range(4)]
    cb = ContextBuilder(context_budget_tokens=10000)
    chunks, citations = [junk] if False else cb.build_context(rs)
    assert [c.num for c in citations] == [1, 2, 3, 4]


def test_noop_reranker_preserves_order():
    from study_llm.rag.reranker import NoOpReranker
    rs = [make_result(chunk_id="a"), make_result(chunk_id="b")]
    out = NoOpReranker().rerank("q", rs)
    assert [r.chunk_id for r in out] == ["a", "b"]


def test_retriever_calls_store_and_provider():
    provider = Mock()
    provider.embed.return_value = [0.1, 0.2]
    store = Mock()
    store.search.return_value = [make_result()]

    from study_llm.rag.retriever import Retriever
    ret = Retriever(provider, vector_store=store, top_k=5)
    out = ret.retrieve("capital of France?")

    provider.embed.assert_called_once_with("capital of France?")
    store.search.assert_called_once()
    assert len(out) == 1


def test_retriever_empty_query_returns_empty():
    provider = Mock()
    store = Mock()
    from study_llm.rag.retriever import Retriever
    ret = Retriever(provider, vector_store=store)
    assert ret.retrieve("   ") == []
    store.search.assert_not_called()


def test_retriever_store_error_returns_empty():
    provider = Mock()
    provider.embed.return_value = [0.1]
    store = Mock()
    store.search.side_effect = RuntimeError("boom")
    from study_llm.rag.retriever import Retriever
    ret = Retriever(provider, vector_store=store)
    assert ret.retrieve("q") == []


def test_pipeline_answer_without_chunks():
    retriever = Mock()
    retriever.retrieve.return_value = []
    inference = Mock()
    inference.generate.return_value = "ignored"

    from study_llm.rag.pipeline import RAGPipeline
    pipe = RAGPipeline(retriever, inference)
    ans = pipe.ask("anything")

    assert ans.answer == "I don't know based on your documents."
    inference.generate.assert_not_called()


def test_pipeline_happy_path():
    retriever = Mock()
    retriever.retrieve.return_value = [make_result()]
    inference = Mock()
    inference.generate.return_value = "Paris. [1]"
    from study_llm.rag.pipeline import RAGPipeline
    pipe = RAGPipeline(retriever, inference)
    ans = pipe.ask("capital of France?")
    assert ans.answer == "Paris. [1]"
    assert ans.chunks_used == 1
    assert len(ans.citations) == 1
    assert "[1]" in ans.answer


def test_pipeline_rerank_applied():
    from study_llm.rag.pipeline import RAGPipeline
    retriever = Mock()
    retriever.retrieve.return_value = [make_result()]
    inference = Mock()
    inference.generate.return_value = "x"
    rr = Mock(spec=["rerank"])
    rr.rerank.return_value = [make_result()]
    pipe = RAGPipeline(retriever, inference, reranker=rr)
    pipe.ask("q")
    rr.rerank.assert_called_once()
