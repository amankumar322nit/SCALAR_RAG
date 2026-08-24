# Scaler Learner Support - Grounded RAG Pipeline & Observability Engine

An end-to-end production Retrieval-Augmented Generation (RAG) system built for Scaler's learner support team. It provides verifiable, grounded answers with source chunk attribution, zero-hallucination guardrails, structured query tracing, and an automated evaluation suite.

---

## 🚀 Key Features

- **Hierarchical Chunking**: Structural markdown parser preserving document hierarchy and contextual section breadcrumbs.
- **Hybrid Dense + Sparse Search**: Combines Cosine Similarity vector embeddings and BM25 Okapi lexical indexing with score fusion.
- **Strict Grounding & Source Citations**: Every factual statement is strictly bound to reference context and attributed via `[Chunk X]` citations.
- **Negative Abstention Guardrails**: Rejects out-of-scope/unsupported queries politely without hallucinating false claims.
- **Structured Observability**: Emits detailed traces to stdout, `data/traces.jsonl`, and SQLite database (`data/traces.db`).
- **Multiple Interfaces**:
  - **CLI**: `python main.py --query "..."` and interactive chat mode `python main.py --interactive`
  - **REST API**: FastAPI service exposing `POST /ask`, `GET /traces`, `POST /ingest`, `GET /health`
  - **Web Chat UI**: Dark glassmorphic interface with interactive source drawers and real-time trace inspector.
- **Automated Evaluation Suite**: Comprehensive benchmark measuring Context Precision, Faithfulness/Grounding, and Fact Recall.

---

## 📁 Repository Structure

```
├── DESIGN.md                          # Exhaustive RAG Architecture & System Design Document
├── README.md                          # Quickstart guide & documentation
├── main.py                            # Root CLI & Server launcher
├── backend/                           # Backend service and core RAG engine
│   ├── main.py                        # Backend service entrypoint
│   ├── requirements.txt               # Backend dependencies
│   ├── .env                           # API keys & configuration
│   ├── corpus/                        # Document corpus (policies, courses, FAQs)
│   ├── src/                           # RAG source code
│   │   ├── config.py                  # Settings & environment variables
│   │   ├── indexer.py                 # Hierarchical parser & chunker
│   │   ├── vector_store.py            # Dense + BM25 hybrid search engine
│   │   ├── llm.py                     # Groq / xAI Grok / OpenAI / Gemini provider
│   │   ├── instrumentation.py         # SQLite & JSONL structured trace logger
│   │   ├── pipeline.py                # RAG pipeline orchestrator
│   │   └── api.py                     # FastAPI endpoints & UI server
│   ├── eval/                          # Evaluation suite
│   │   ├── test_cases.json            # 10 benchmark test cases
│   │   ├── run_eval.py                # Automated evaluation runner
│   │   └── eval_report.json           # Evaluation benchmark report
│   └── data/                          # SQLite traces.db and vector index cache
└── frontend/                          # Web UI client
    ├── index.html                     # Responsive single-page interface
    ├── style.css                      # Modern dark theme & scrollbar styles
    └── app.js                         # Chat controller, citation tags & trace inspector
```

---

## ⚡ Quickstart

### 1. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# OR
pip install fastapi uvicorn requests python-dotenv pydantic rich numpy
```

*(Optional)* To use remote LLM generation (e.g. Grok / xAI, OpenAI, or Gemini), set your API key in `.env`:
```bash
echo "GROK_API_KEY=your_xai_grok_key_here" >> .env
# or
echo "OPENAI_API_KEY=your_openai_key_here" >> .env
# or
echo "GEMINI_API_KEY=your_gemini_key_here" >> .env
```
*Note: The pipeline supports Grok (`grok-2-latest`, `grok-beta`), OpenAI, Gemini, and falls back to a high-speed offline grounded synthesizer if no key is supplied.*

---

### 2. Run via CLI

#### Single Query:
```bash
python main.py --query "What is the 7-day refund policy?"
```

#### Interactive Chat:
```bash
python main.py --interactive
```

#### Reindex Corpus:
```bash
python main.py --ingest
```

---

### 3. Run Web UI & REST API Server

```bash
python main.py --server --port 8000
```
- Open browser at: **`http://localhost:8000`**
- API documentation: **`http://localhost:8000/docs`**

#### Sample API Request:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the eligibility criteria for placement support?"}'
```

---

### 4. Run Automated Evaluation Suite

```bash
python eval/run_eval.py
```

**Benchmark Results:**
- **Test Suite Pass Rate**: **100.0%** (10/10 test cases passed)
- **Context Precision (Retrieval Hit Rate)**: **100.0%**
- **Average Fact Recall**: **96.3%**
- **Faithfulness & Grounding Score**: **100.0%**
- **Average End-to-End Latency**: **~1.7 ms**
