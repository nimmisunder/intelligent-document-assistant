# 📄 Intelligent Document Assistant

An AI-powered document question-answering system that allows users to upload PDF documents, extract and index their content, and ask questions using Retrieval-Augmented Generation (RAG).

The application combines semantic search, keyword search, hybrid retrieval, and Cross-Encoder reranking to retrieve relevant information from documents. Ollama LLM generates context-aware answers with source references and PDF page numbers.

## 🚀 Features

### 1. Document Management
- Upload PDF documents.
- Extract text and page numbers using PyMuPDF.
- Automatically split documents into meaningful chunks.
- Store document metadata in SQLite.
- View uploaded documents.
- Delete documents and their associated indexes.

### 2. Intelligent RAG Pipeline
- **Text Chunking:** Split documents into smaller, meaningful chunks.
- **Embeddings:** Generate semantic embeddings using Sentence Transformers.
- **Vector Search:** Store and retrieve embeddings using FAISS.
- **Keyword Search:** Retrieve relevant content using BM25.
- **Hybrid Search:** Combine semantic and keyword retrieval using Reciprocal Rank Fusion (RRF).
- **Reranking:** Improve retrieval relevance using a Cross-Encoder.

### 3. AI-Powered Chatbot
- Ask natural-language questions about uploaded PDFs.
- Generate answers using a locally hosted Ollama LLM.
- Retrieve relevant document context before generating answers.
- Display source document names and page numbers.
- Support searching across all documents or a selected document.
- Maintain chat history within the Streamlit session.

### 4. RAG Evaluation Dashboard
Evaluate retrieval quality using:

| Metric | Description |
|---|---|
| Precision@5 | Measures the proportion of retrieved chunks that are relevant. |
| Recall@5 | Measures the proportion of relevant chunks successfully retrieved. |
| MRR | Measures how highly the first relevant result is ranked. |
| Hit Rate@5 | Measures whether at least one relevant chunk was retrieved. |

The evaluation dashboard displays overall metrics, charts, and individual question results.

---

## 🏗️ System Architecture

```text
                 USER
                   |
                   v
          Streamlit Frontend
                   |
                   v
              FastAPI
                   |
          +--------+--------+
          |                 |
          v                 v
    Document Upload      AI Chat
          |                 |
          v                 v
      PDF Extraction    User Question
          |                 |
          v                 v
       Chunking        Query Embedding
          |                 |
          v                 v
      Embeddings       Hybrid Retrieval
          |             /         \
          v            v           v
         FAISS        FAISS       BM25
          |              \         /
          |               v       v
          |                  RRF
          |                   |
          |                   v
          |              Reranking
          |                   |
          |                   v
          |             Relevant Context
          |                   |
          |                   v
          |              Ollama LLM
          |                   |
          |                   v
          |              AI Answer
          |                   |
          +-------------------+
                              |
                              v
                    Answer + Citations
                              |
                              v
                       Streamlit UI
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| LLM | Ollama |
| LLM Model | Llama 3 |
| Database | SQLite |
| Evaluation | Custom retrieval metrics |

## 📂 Project Structure

```text
intelligent-document-assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   ├── chat.py
│   │   └── evaluation.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── pdf_service.py
│       ├── chunking_service.py
│       ├── embedding_service.py
│       ├── vector_store.py
│       ├── bm25_service.py
│       ├── hybrid_search.py
│       ├── reranking_service.py
│       ├── indexing_service.py
│       ├── llm_service.py
│       └── evaluation_service.py
│
├── frontend/
│   └── main.py
│
├── evaluation/
│   ├── evaluation_dataset.json
│   └── evaluation_results.json
│
├── uploads/
│
├── vector_store/
│   ├── faiss.index
│   └── metadata.json
│
├── evaluate_retrieval.py
├── test_chunking.py
├── test_ingestion.py
├── test_faiss.py
├── test_bm25.py
├── test_hybrid.py
├── test_reranking.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Ollama installed.
- Git.
- A compatible local LLM model.
- Recommended: 16 GB RAM for comfortable local development, depending on model size.

### 1. Clone the repository

```bash
git clone https://github.com/nimmisunder/intelligent-document-assistant.git

cd intelligent-document-assistant
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If you don't have a requirements file yet, install:

```powershell
pip install fastapi uvicorn python-multipart pymupdf streamlit requests pandas numpy sentence-transformers faiss-cpu rank-bm25 langchain-text-splitters
```

### 4. Configure Ollama

Install Ollama and download the Llama 3 model:

```powershell
ollama pull llama3
```

Verify the model:

```powershell
ollama list
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

The application uses
