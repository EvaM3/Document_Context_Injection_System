# RAG Document Analysis System

AI-powered Retrieval-Augmented Generation (RAG) system for analyzing PDF and text documents using semantic search, embeddings, and vector retrieval.

## Features

- PDF and TXT document ingestion
- Recursive document chunking
- HuggingFace sentence embeddings
- Semantic vector retrieval
- Context-aware question answering
- Source tracking for retrieved documents
- Google Gemini integration via LangChain

## Tech Stack

- Python
- LangChain
- Google Gemini
- HuggingFace Embeddings
- InMemoryVectorStore
- PyPDF
- Sentence Transformers

## Architecture

```text
Documents
   ↓
Loaders (PDF/TXT)
   ↓
Text Chunking
   ↓
Embeddings
   ↓
Vector Store
   ↓
Similarity Search
   ↓
Retrieved Context
   ↓
Gemini LLM Response
```

## Example Use Cases

- Financial document analysis
- Knowledge base search
- AI-powered document assistants
- Semantic document retrieval
- Internal company documentation querying

## Example Queries

```text
Summarize the balance sheet.
```

```text
Which balance sheet items increased from 2023 to 2024?
```

```text
Compare short-term investments across both years.
```

## Project Structure

```text
.
├── rag.py
├── main.py
├── My Documents/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd Document_Context_Injection_System
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python rag.py
```

## Future Improvements

- FAISS vector database integration
- Conversational memory
- Streamlit web interface
- Metadata filtering
- Multi-query retrieval
- Persistent vector storage
- API deployment

## Author

Eva Madaraszova
