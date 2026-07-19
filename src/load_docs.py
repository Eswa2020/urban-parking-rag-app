"""
load_docs.py
Loads raw documents (PDF, TXT) from data/documents/<city>/ into LangChain
Document objects, tagging each with 'city' metadata based on its subfolder.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

DOCUMENTS_DIR = os.path.join("data", "documents")


def load_documents(data_dir: str = DOCUMENTS_DIR):
    """
    Walk each city subfolder under data_dir, load supported files, and
    return a flat list of Document objects tagged with 'city' metadata.
    """
    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"'{data_dir}' does not exist. Create it and add city subfolders first."
        )

    city_folders = [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ]

    if not city_folders:
        raise FileNotFoundError(
            f"No subfolders found in '{data_dir}'. Expected e.g. dubai/, seattle/, methodology/."
        )

    for city in city_folders:
        city_dir = os.path.join(data_dir, city)

        for filename in os.listdir(city_dir):
            filepath = os.path.join(city_dir, filename)

            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                print(f"Skipping unsupported file: {filepath}")
                continue

            loaded = loader.load()

            for doc in loaded:
                doc.metadata["city"] = city

            print(f"Loaded {len(loaded)} page(s)/section(s) from {city}/{filename}")
            documents.extend(loaded)

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"\nTotal document chunks loaded: {len(docs)}")

    if docs:
        print("\n--- Sample from first document ---")
        print("City:", docs[0].metadata.get("city"))
        print("Source:", docs[0].metadata.get("source"))
        print("Content preview:", docs[0].page_content[:300])