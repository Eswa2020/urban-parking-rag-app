# Urban Parking RAG App

## Overview
This repository contains a Streamlit-based decision-support application for urban parking analysis. The app is designed to combine spatial machine learning outputs with retrieval-augmented generation (RAG) so that users can explore parking imbalance patterns and receive source-grounded planning or policy guidance.

The application is based on an urban parking use case developed from a master's research project focused on parking accessibility, spatial demand patterns, and decision support.

## Project Goal
The goal of this app is to make urban parking analysis more interpretable and actionable by connecting model outputs with relevant policy or planning documents through a simple interactive interface.

## Core Features
- interactive Streamlit interface
- question-answering over parking and policy documents
- retrieval of relevant planning or transport guidance
- source-grounded responses using RAG
- integration path for ML outputs and zone-level summaries

## Planned Workflow
1. load parking-related policy and planning documents
2. split and embed text into a searchable vector store
3. retrieve relevant content based on user queries
4. generate grounded answers with citations or referenced sources
5. display responses in a Streamlit app
6. optionally connect model outputs from the urban parking intelligence project

## Tech Stack
- Python
- Streamlit
- LangChain or LlamaIndex
- Vector store such as FAISS or Chroma
- OpenAI-compatible or local embeddings
- Optional integration with spatial ML outputs

## Repository Structure
```text
urban-parking-rag-app/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── data/
│   ├── documents/
│   └── processed/
├── src/
│   ├── load_docs.py
│   ├── chunking.py
│   ├── embed_store.py
│   ├── retriever.py
│   ├── generate.py
│   └── utils.py
├── outputs/
│   └── screenshots/
└── notebooks/
    └── rag_experiments.ipynb
