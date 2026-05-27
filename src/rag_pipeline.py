"""
RAG Pipeline — Retrieval-Augmented Generation core.
Retrieves relevant KPI context, constructs prompt, returns GPT-4 response.
Mirrors production pipeline: Azure AI Search retrieval + Azure OpenAI generation.
"""

import os
from typing import List, Dict

from openai import OpenAI
from src.vector_store import search

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


SYSTEM_PROMPT = """You are a healthcare operations analytics assistant for a multi-doctor optometry clinic.
You have access to real-time KPI data for each doctor and department.
Answer questions accurately and concisely using ONLY the provided context.
If the data doesn't contain what was asked, say so clearly — do not hallucinate metrics.
Format numbers and trends clearly. Keep responses under 150 words unless a detailed breakdown is requested."""


def build_context(retrieved_docs: List[Dict]) -> str:
    """Format retrieved documents into context block for the prompt."""
    context_lines = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_lines.append(
            f"[{i}] {doc['doctor']} | {doc['department']} | "
            f"{doc['metric']}: {doc['value']} ({doc['trend']}) — {doc['period']}\n"
            f"    Note: {doc['notes']}"
        )
    return "\n\n".join(context_lines)


def query(user_question: str, top_k: int = 3, chat_history: List[Dict] = None) -> Dict:
    """
    Full RAG query: retrieve → augment → generate.

    Args:
        user_question: Natural language question from user
        top_k: Number of documents to retrieve
        chat_history: Prior conversation turns for multi-turn support

    Returns:
        Dict with 'answer', 'sources', and 'retrieved_docs'
    """
    # Step 1 — Retrieve
    retrieved_docs = search(user_question, top_k=top_k)
    context = build_context(retrieved_docs)

    # Step 2 — Augment prompt
    user_message = f"""Context (KPI Data):\n{context}\n\nQuestion: {user_question}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include chat history for multi-turn conversation
    if chat_history:
        messages.extend(chat_history[-6:])  # Last 3 turns only (token efficiency)

    messages.append({"role": "user", "content": user_message})

    # Step 3 — Generate
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,  # Low temp for factual KPI responses
        max_tokens=300,
    )

    answer = response.choices[0].message.content

    sources = [
        f"{doc['doctor']} — {doc['metric']} ({doc['period']})"
        for doc in retrieved_docs
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_docs": retrieved_docs,
        "tokens_used": response.usage.total_tokens,
    }
