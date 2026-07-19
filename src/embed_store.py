"""
embed_store.py
Embeds document chunks and stores them in a local FAISS vector index.
Defaults to OpenAI embeddings; falls back to a local sentence-transformers
model if no OPENAI_API_KEY is set.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from chunking import chunk_documents
from load_docs import load_documents

load_dotenv()

VECTORSTORE_DIR = os.path.join("data", "processed", "vectorstore")


def get_embeddings():
    """
    Return an embeddings object: OpenAI if an API key is available,
    otherwise a local sentence-transformers model.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        from langchain_openai import OpenAIEmbeddings
        print("Using OpenAI embeddings (text-embedding-3-small).")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("No OPENAI_API_KEY found — using local sentence-transformers embeddings.")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vectorstore(chunks, save_path: str = VECTORSTORE_DIR):
    """
    Embed the given chunks and save them as a FAISS index at save_path.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"Vector store saved to: {save_path}")

    return vectorstore


def load_vectorstore(save_path: str = VECTORSTORE_DIR):
    """
    Load a previously saved FAISS index from disk.
    """
    embeddings = get_embeddings()
    return FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Embedding {len(chunks)} chunk(s)...")

    build_vectorstore(chunks)