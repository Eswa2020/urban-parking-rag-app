"""
generate.py
Takes retrieved chunks + a user query, builds a grounded prompt, and
generates an answer via OpenAI or Ollama (switchable via LLM_PROVIDER in .env).
"""

import os
from dotenv import load_dotenv

from retriever import retrieve

load_dotenv()

SYSTEM_PROMPT = """You are a planning and policy assistant for an urban parking \
decision-support tool. Answer the user's question using ONLY the provided context \
chunks below. Each chunk includes a source and page number — cite them inline \
using the format (source, p.PAGE) after any claim drawn from that chunk.

If the context does not contain enough information to answer confidently, say so \
explicitly rather than guessing or inventing details. Do not state anything as \
regulation or policy fact unless it is directly supported by the context."""


def get_llm():
    """
    Return a chat model: OpenAI or Ollama, based on LLM_PROVIDER in .env.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        print(f"Using Ollama ({model_name}).")
        return ChatOllama(model=model_name, temperature=0.2)
    else:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI ({model_name}).")
        return ChatOpenAI(model=model_name, temperature=0.2)


def build_context(chunks):
    """
    Format retrieved chunks into a labeled context block for the prompt.
    """
    parts = []
    for doc in chunks:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, k: int = 4, city: str = None):
    """
    Retrieve relevant chunks, build a grounded prompt, and return the LLM's answer.
    """
    chunks = retrieve(query, k=k, city=city)
    context = build_context(chunks)

    llm = get_llm()

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)
    return response.content, chunks


if __name__ == "__main__":
    test_query = "What are the minimum off-street parking requirements for retail uses?"
    print(f"Query: {test_query}\n")

    answer, sources = generate_answer(test_query)

    print("--- Answer ---")
    print(answer)

    print("\n--- Sources used ---")
    for doc in sources:
        print(
            "-", doc.metadata.get("city", "unknown"),
            "|", os.path.basename(doc.metadata.get("source", "unknown")),
            "p.", doc.metadata.get("page")
        )