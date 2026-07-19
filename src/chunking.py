"""
chunking.py
Splits loaded documents (page-level) into smaller, overlapping chunks
suitable for embedding and retrieval.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_docs import load_documents

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks


def chunk_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Take a list of page-level LangChain Documents and split them into
    smaller chunks, preserving source metadata on each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} page(s).")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")

    print("\n--- Sample chunk ---")
    print("Source:", chunks[0].metadata.get("source"))
    print("Page:", chunks[0].metadata.get("page"))
    print("Content:", chunks[0].page_content)