"""
retriever.py
Loads the saved FAISS vector store and retrieves the top-k most
relevant chunks for a given query, with optional city filtering.
"""

from embed_store import load_vectorstore

DEFAULT_K = 4  # number of chunks to retrieve per query


def retrieve(query: str, k: int = DEFAULT_K, city: str = None):
    """
    Run a query against the vector store, return the top-k matching chunks.
    If city is given, only chunks tagged with that city are considered.
    """
    vectorstore = load_vectorstore()

    if city and city.lower() != "all":
        candidates = vectorstore.similarity_search(query, k=k * 5)
        filtered = [doc for doc in candidates if doc.metadata.get("city") == city.lower()]
        return filtered[:k]
    else:
        return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    test_query = "What are the minimum off-street parking requirements for retail uses?"
    print(f"Query: {test_query}\n")

    results = retrieve(test_query, city="columbus")

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("City:", doc.metadata.get("city"))
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print("Content:", doc.page_content[:300])
        print()