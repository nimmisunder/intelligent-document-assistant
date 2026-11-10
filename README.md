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

The application uses the local Ollama API:

http://localhost:11434

The configured model name is:

llama3:latest

5. Start FastAPI

From the project root:

uvicorn app.main:app --reload

FastAPI runs at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

6. Start Streamlit

Open another terminal:

streamlit run frontend/main.py

The application will open in your browser.

🔄 How the RAG Pipeline Works

Step 1: Document Ingestion

Users upload PDF documents through the Streamlit interface.

PyMuPDF extracts the document text page by page while preserving page-number metadata.

Step 2: Chunking

Extracted text is split into smaller chunks using RecursiveCharacterTextSplitter.

Initial configuration:

Chunk size: 500 characters.

Chunk overlap: 100 characters.

Each chunk stores metadata such as document ID, filename, page number, and chunk ID.

Step 3: Embedding Generation

Sentence Transformers converts text chunks into numerical vectors.

Model:

all-MiniLM-L6-v2

Embedding dimension: 384.

Step 4: Semantic Search

FAISS stores the vectors and retrieves semantically similar chunks based on the user's question.

Step 5: Keyword Search

BM25 retrieves chunks based on keyword relevance.

Step 6: Hybrid Retrieval

Reciprocal Rank Fusion combines the rankings from FAISS and BM25.

Step 7: Reranking

A Cross-Encoder evaluates the question and retrieved passages together to improve relevance ranking.

Model:

cross-encoder/ms-marco-MiniLM-L-6-v2

Step 8: Answer Generation

The retrieved passages are passed to Ollama along with the user's question.

The LLM generates an answer using the retrieved document context.

Step 9: Source References

The application returns source metadata, including filenames and PDF page numbers, to help users identify where the information originated.

📊 Evaluation

The application uses a labelled evaluation dataset.

Example:

[
    {
        "question": "What health insurance benefits are provided?",
        "document_id": "YOUR_DOCUMENT_ID",
        "relevant_chunk_ids": [
            "YOUR_RELEVANT_CHUNK_ID"
        ]
    }
]

Run evaluation:

python evaluate_retrieval.py

The results are saved under:

evaluation/evaluation_results.json

The Streamlit dashboard displays the calculated retrieval metrics.

Note: Evaluation results depend on the quality and completeness of the manually labelled ground-truth dataset.

🔐 Data Storage

The application uses:

SQLite for document metadata.

Local PDF storage in uploads/.

FAISS index storage in vector_store/.

JSON metadata for indexed chunks.

In-memory BM25 index rebuilt from persisted chunk metadata.

Uploaded documents and indexes are stored locally.

Do not commit private documents, database files, API secrets, or generated indexes to a public repository.

🧪 Example Questions

Users can ask questions such as:

What benefits are mentioned in this document?

What are the eligibility criteria?

What are the exclusions?

Which page mentions a particular benefit?

Summarize the important points of this policy.

The actual answers depend on the contents of the uploaded documents.

🚧 Future Enhancements

Compare FAISS, BM25, Hybrid Search, and Hybrid + Reranking.

Integrate Ragas for answer-quality evaluation.

Add faithfulness and answer relevancy metrics.

Add conversation persistence.

Add PDF preview with page navigation.

Add document-level access control.

Improve handling of scanned PDFs using OCR.

Add streaming LLM responses.

Improve indexing reliability and background processing.

Add deployment using Docker.

🎯 Project Objective

This project demonstrates practical implementation of Retrieval-Augmented Generation, semantic search, hybrid retrieval, reranking, local LLM integration, and retrieval evaluation.

It explores how AI-powered systems can retrieve relevant information from private documents and generate context-aware responses.


Technologies explored: Python | FastAPI | Streamlit | RAG | FAISS | BM25 | Sentence Transformers | Ollama | LLM Evaluation
