# RAG PDF Assistant

An agentic RAG system that lets you upload a PDF and have a conversation with it. Built with FastAPI, LangGraph, Gemini 2.5 Flash, hybrid search (BM25 + vector), and a Streamlit UI.

---

## Features

- Upload and index PDF documents
- Agentic pipeline: query routing, document grading, query rewriting, hallucination checking
- Hybrid search: BM25 + semantic vector search fused with Reciprocal Rank Fusion
- Cross-encoder reranking for precision
- Conversation memory per session
- Streaming responses via Server-Sent Events
- Streamlit chat UI
- RAGAS evaluation suite
- Automated tests with pytest
- CI via GitHub Actions

---

## Architecture

```
User question
     │
     ▼
LangGraph Agent
  ├── Router          (retrieval vs. direct response)
  ├── Retrieve        (hybrid search + cross-encoder rerank)
  ├── Grade Docs      (relevance filter)
  ├── Rewrite Query   (retry if no relevant docs)
  ├── Generate        (Gemini 2.5 Flash + chat history)
  └── Hallucination   (grounded check, retry if needed)
```

---

## Project Structure

```
rag-pdf-assistant/
├── app/
│   ├── main.py          # FastAPI routes + SSE streaming endpoint
│   └── storage.py       # doc_id generation, file paths
├── rag/
│   ├── agents/          # LangGraph nodes (router, grader, generator, …)
│   ├── chains/          # generation, retrieval, rerank chains
│   ├── ingest.py        # PDF parsing, chunking, indexing
│   ├── llm.py           # Gemini LLM + HuggingFace embeddings
│   └── store.py         # ChromaDB interface
├── ui/
│   ├── streamlit_app.py
│   └── components/      # sidebar, chat
├── eval/
│   ├── test_queries.json  # sample Q&A pairs for evaluation
│   ├── ragas_eval.py      # RAGAS evaluation logic
│   └── run_eval.py        # CLI runner + results summary
├── tests/               # pytest suite
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

---

## Installation

### 1. Clone

```bash
git clone https://github.com/robayedl/rag-pdf-assistant.git
cd rag-pdf-assistant
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

---

## Running

### Start the backend (port 8000)

```bash
uvicorn app.main:app --reload --port 8000
```

### Start the Streamlit UI (port 8501)

```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running with Docker

```bash
docker compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/documents` | Upload a PDF |
| POST | `/documents/{doc_id}/index` | Index a document |
| POST | `/query` | Ask a question (full response) |
| POST | `/query/stream` | Ask a question (SSE streaming) |

---

## Running Tests

```bash
pytest -q
```

---

## RAGAS Evaluation

Evaluate the RAG pipeline quality using [RAGAS](https://docs.ragas.io) metrics: faithfulness, answer relevancy, context precision, and context recall.

### 1. Upload and index a PDF via the API

```bash
# Upload
DOC_ID=$(curl -s -F "file=@your_document.pdf" http://localhost:8000/documents | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_id'])")

# Index
curl -s -X POST http://localhost:8000/documents/$DOC_ID/index
```

### 2. Edit the test queries (optional)

Open `eval/test_queries.json` and replace `"REPLACE_WITH_YOUR_DOC_ID"` with your actual `doc_id`, and update the `ground_truth` values to match your document.

### 3. Run evaluation

```bash
# Using the --doc-id flag to override all queries at once:
python eval/run_eval.py --doc-id <your_doc_id>
```

### 4. View results

Results are saved to `eval/results.json` with a timestamp. The latest scores also appear in the Streamlit sidebar under **Evaluation Scores**.

**Pass criteria:** `faithfulness ≥ 0.7` and `answer_relevancy ≥ 0.7`

---

## License

This project is provided for educational and research purposes.
