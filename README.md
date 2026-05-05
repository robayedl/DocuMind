# DocuMind — Agentic Document Intelligence

> Chat with any PDF using a production-grade agentic pipeline powered by LangGraph, Gemini 2.5 Flash, hybrid search, and real-time streaming.

[![CI](https://github.com/robayedl/documind/actions/workflows/ci.yml/badge.svg)](https://github.com/robayedl/documind/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Demo

https://github.com/user-attachments/assets/290e7caf-6676-43c2-9f9f-9df63d28c3f9

---

## Features

| Feature | Description |
|---|---|
| **Agentic RAG** | LangGraph pipeline with routing, grading, rewriting, and hallucination checking |
| **Hybrid Search** | BM25 + semantic vector search fused with Reciprocal Rank Fusion (RRF) |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` reranker for high-precision results |
| **Gemini 2.5 Flash** | Google's fastest frontier LLM for low-latency answers |
| **Streaming Responses** | Server-Sent Events (SSE) for real-time token-by-token output |
| **Conversation Memory** | Per-session chat history maintained across turns |
| **RAGAS Evaluation** | Faithfulness, answer relevancy, context precision & recall |
| **Streamlit Chat UI** | Dark-theme UI with PDF viewer, SSE streaming, and source citations |
| **Docker Ready** | Full multi-service docker-compose setup (API + UI) |
| **AWS EC2 Deployment** | Production-ready for cloud deployment |

---

## Architecture

```mermaid
flowchart TD
    Q([User Question]) --> R[Router]

    R -->|direct| DA[Direct Answer]
    DA --> E1([END])

    R -->|retrieve| RET[Retrieve]
    RET --> GD[Grade Docs]

    GD -->|relevant docs| GEN[Generate]
    GD -->|no docs · retry < 3| RW[Rewrite Query]
    GD -->|no docs · max retries| FB[Fallback]

    RW --> RET

    GEN --> HC[Hallucination Check]

    HC -->|grounded| RESP([Final Response + Citations])
    HC -->|not grounded · retry < 3| GEN
    HC -->|max retries| FB

    FB --> E2([END])
```

**Retrieval Pipeline:**
```
Query ──► BM25 Sparse Search  ─┐
                                ├──► RRF Fusion ──► Cross-Encoder Rerank ──► Top-K Chunks
Query ──► Vector Dense Search ─┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.115, Uvicorn, Server-Sent Events |
| **Agent Framework** | LangGraph (StateGraph), LangChain |
| **LLM** | Google Gemini 2.5 Flash (`langchain-google-genai`) |
| **Embeddings** | HuggingFace `all-mpnet-base-v2` (768-dim) |
| **Vector Store** | ChromaDB 0.5 |
| **Sparse Search** | BM25 (`rank-bm25`) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **PDF Parsing** | pdfplumber + Unicode normalization |
| **Chunking** | RecursiveCharacterTextSplitter (800 tokens, 100 overlap) |
| **UI** | Streamlit (dark theme, SSE streaming, PDF viewer) |
| **Evaluation** | RAGAS (faithfulness, answer relevancy, context precision/recall) |
| **Testing** | pytest + ruff |
| **CI/CD** | GitHub Actions |
| **Container** | Docker + docker-compose |

---

## Quick Start

### Option 1 — Docker (Recommended)

**1. Clone the repo**
```bash
git clone https://github.com/robayedl/documind.git
cd documind
```

**2. Set your API key**
```bash
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_key_here
```

**3. Launch everything**
```bash
docker compose up --build
```

- **UI:** [http://localhost:8501](http://localhost:8501)
- **API:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2 — Local Development

**1. Clone & enter directory**
```bash
git clone https://github.com/robayedl/documind.git
cd documind
```

**2. Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```
> Windows: `.venv\Scripts\activate`

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment**
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY
```

**5. Start backend** _(terminal 1)_
```bash
uvicorn app.main:app --reload --port 8000
```

**6. Start UI** _(terminal 2)_
```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

---

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns status and environment |
| `POST` | `/documents` | Upload a PDF file, returns `doc_id` |
| `POST` | `/documents/{doc_id}/index` | Parse, chunk, and index a document |
| `GET` | `/documents/{doc_id}/file` | Download the original PDF |
| `POST` | `/query` | Ask a question, get a full JSON response |
| `POST` | `/query/stream` | Ask a question, receive SSE streaming tokens |

### Example: Upload and query a PDF

**Upload**
```bash
DOC_ID=$(curl -s -F "file=@document.pdf" http://localhost:8000/documents \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_id'])")
```

**Index**
```bash
curl -s -X POST http://localhost:8000/documents/$DOC_ID/index
```

**Query (full response)**
```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"$DOC_ID\", \"question\": \"What is this document about?\"}"
```

**Query (streaming)**
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"$DOC_ID\", \"question\": \"Summarize the key points.\"}"
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** Google AI Studio API key |
| `STORAGE_DIR` | `./storage` | Directory for uploaded PDFs |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed origins |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL for Streamlit UI |

---

## Running Tests

**Run full test suite**
```bash
pytest -q
```

**Run with coverage**
```bash
pytest --cov=rag --cov=app -q
```

**Lint**
```bash
ruff check .
```

---

## Evaluation

DocuMind ships with a 30-question golden dataset built from **"Attention Is All You Need"** (Vaswani et al., 2017) and an automated [RAGAS](https://docs.ragas.io) evaluation runner that uses **Gemini 2.5 Flash** as the judge model.

### Latest Evaluation Results

<!-- EVAL-RESULTS-START -->
| Metric | Score | |
|---|---|---|
| `faithfulness` | 0.879 | █████████████████ |
| `answer_relevancy` | 0.706 | ██████████████ |
| `context_precision` | 0.872 | █████████████████ |
| `context_recall` | 0.883 | █████████████████ |

_Evaluated on 30 questions · 2026-05-05 · full results in [`eval/results/latest.json`](eval/results/latest.json)_
<!-- EVAL-RESULTS-END -->

### Metrics

| Metric | What it measures |
|---|---|
| `faithfulness` | Are all claims in the answer grounded in the retrieved context? |
| `answer_relevancy` | Is the answer actually relevant to the question? |
| `context_precision` | Are the most relevant chunks ranked highest? |
| `context_recall` | Does the context cover everything needed to answer the question? |

### Running the evaluation

**1. Start the server, upload, and index the PDF** _(server only needed for this step)_
```bash
uvicorn app.main:app --reload --port 8000
```
```bash
DOC_ID=$(curl -s -F "file=@attention.pdf" http://localhost:8000/documents \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_id'])")
curl -s -X POST http://localhost:8000/documents/$DOC_ID/index
```
> Once indexed, the server can be stopped — `make eval` calls the pipeline directly.

**2. Run full evaluation**
```bash
DOC_ID=$DOC_ID make eval
```
> ~10–15 min · ~$0.05–$0.15 in Gemini API calls (30 questions × ~5 LLM calls each for pipeline + RAGAS scoring)

**Quick smoke-test** _(5 questions)_
```bash
python eval/run.py --doc-id $DOC_ID --limit 5
```

**Filter by category**
```bash
python eval/run.py --doc-id $DOC_ID --category factual
```

After each run the scores above are automatically updated in this file. To refresh them from the latest result without re-running evaluation:

```bash
make update-readme
```

See [eval/EVALUATION_GUIDE.md](eval/EVALUATION_GUIDE.md) for dataset format, how to add entries, and cost estimates.

---

## Project Structure

```
documind/
├── app/
│   ├── main.py              # FastAPI app — all routes + SSE streaming
│   └── storage.py           # doc_id generation, file path helpers
│
├── rag/
│   ├── agents/
│   │   ├── graph.py         # LangGraph StateGraph — full pipeline
│   │   ├── router.py        # Query router (retrieve vs. direct)
│   │   ├── grader.py        # Document relevance grader
│   │   ├── generator.py     # Answer generator (Gemini + chat history)
│   │   ├── hallucination.py # Groundedness checker
│   │   ├── rewriter.py      # Query rewriter for retry loop
│   │   └── memory.py        # Per-session conversation memory
│   ├── chains/
│   │   ├── retrieval.py     # Hybrid search + RRF fusion
│   │   └── rerank.py        # Cross-encoder reranker
│   ├── ingest.py            # PDF parsing (pdfplumber), chunking, indexing
│   ├── llm.py               # Gemini LLM + HuggingFace embeddings
│   └── store.py             # ChromaDB interface (collection versioning)
│
├── ui/
│   ├── streamlit_app.py     # Main Streamlit entrypoint
│   └── components/
│       ├── chat.py          # Chat interface + SSE streaming
│       ├── sidebar.py       # Upload, index, document management
│       └── pdf_viewer.py    # Inline PDF viewer
│
├── eval/
│   ├── golden.jsonl         # 30-question golden dataset (Attention Is All You Need)
│   ├── run.py               # RAGAS runner — per-question table, JSON output, README update
│   ├── results/             # Per-run result JSONs (<UTC-timestamp>.json)
│   └── EVALUATION_GUIDE.md  # Dataset format, field definitions, usage, cost estimates
│
├── tests/
│   ├── test_agent.py        # Agent node + graph integration tests
│   └── test_llm_pipeline.py # LLM init + embeddings tests
│
├── .env.example             # Environment variable template
├── .github/workflows/ci.yml # CI: lint + test on push/PR
├── Dockerfile               # Python 3.12 slim image
├── docker-compose.yml       # Multi-service: api + streamlit
├── Makefile                 # run / ui / test / lint / eval targets
└── requirements.txt
```

---

## License

MIT — free to use, modify, and distribute.
