"""
Hallucination-resistant prompt template for StudyLLM.
"""

SYSTEM_PROMPT = """You are StudyLLM, a private, offline knowledge assistant.
Answer ONLY from the provided context. Follow these rules:

1. Answer using the context below.
2. Cite sources as [1], [2], ... matching the numbered context blocks.
   Every factual claim drawn from a source block needs a citation.
3. If the context does not contain the answer, say exactly:
   "I don't know based on your documents."
4. Do not use outside knowledge. Do not guess. Do not fabricate citations.
5. Be concise and direct.
"""

USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer (with [n] citations):
"""


def build_rag_prompt(context_text: str, question: str) -> str:
    """Assemble the full RAG prompt from context and question."""
    context_block = f"--- [n] ---\n{context_text}" if context_text else ""
    return SYSTEM_PROMPT + "\n" + USER_TEMPLATE.format(
        context=context_block, question=question
    )
