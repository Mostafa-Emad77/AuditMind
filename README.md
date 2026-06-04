# AuditMind

**Bilingual agentic financial document auditor for Arabic/English workflows**

AuditMind analyzes financial PDFs in Arabic and English, cross-checks them with hybrid RAG (vector + graph), and streams agent reasoning to the UI in real time. It targets Egyptian and MENA use cases with mixed-language documents.

---

## What it does

Upload invoices, contracts, bank statements, or balance sheets (scanned or digital). AuditMind:

1. **Extracts** text — PyMuPDF for digital PDFs; EasyOCR (multilingual) + ArabicOCR fallback for scans  
2. **Embeds** chunks into **Qdrant** (sentence-transformers, 768-dim) and builds an entity graph in **Neo4j**  
3. **Plans** a targeted audit checklist from the detected document types  
4. **Cross-checks** across documents — 4-phase: graph contradiction detection → LLM adjudication → checklist-driven semantic retrieval → deduplication  
5. **Surfaces contradictions** (amount mismatches, date inconsistencies, party discrepancies) with severity and confidence  
6. **Reconciles** contract totals, invoice totals, and bank payments into a snapshot with variance  
7. **Writes** a structured report in Arabic or English  
8. **Streams** every agent reasoning step over **SSE** to the Next.js UI  

---

## Architecture

```
PDFs
  → Ingestion (EasyOCR / PyMuPDF / ArabicOCR)
      → Qdrant (chunks, embeddings)  +  Neo4j (entities & relations)
  → Planner → Cross-checker (hybrid RAG + graph tools + web search)
      → Report writer
  → FastAPI (SSE) → Next.js UI
```

**Infra (Docker Compose):** Redis (sessions + reports, 7-day TTL), Qdrant, FastAPI backend, Next.js frontend. **Neo4j** is external (e.g. [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/)) — configure `NEO4J_*` in `backend/.env`.

**Performance:** multi-document uploads are ingested (OCR + embedding) concurrently in a thread pool sized to `os.cpu_count()`. Entity extraction runs per-document in parallel with `asyncio.gather`. Hybrid-RAG query routing and checklist-intent classification are rule-based (no per-item LLM calls). All cross-checker LLM calls use `ainvoke` so they don't block the event loop or SSE keepalives.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Agents | LangGraph 0.2 (`StateGraph`) |
| LLM | [OpenRouter](https://openrouter.ai/) (default) or [Google AI Studio](https://aistudio.google.com/) (Gemini) — set `LLM_PROVIDER` |
| Embeddings | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim, multilingual) |
| OCR | EasyOCR (multilingual), ArabicOCR fallback (Python <3.12), PyMuPDF for digital PDFs |
| Vector DB | Qdrant |
| Graph DB | Neo4j (Aura or self-hosted) |
| Cache / sessions | Redis |
| Backend | FastAPI, SSE, Python 3.11+ |
| Frontend | Next.js 15, React 19, Tailwind CSS, Framer Motion, Radix UI |
| Ops | Docker, Docker Compose |

---

## Prerequisites

- **Python 3.11+** (matches `backend/Dockerfile`)  
- **Node.js 20+**  
- **Docker** with Compose v2 (`docker compose`)  
- **Neo4j** reachable from the backend (Aura free tier works)  
- **OpenRouter API key** — or a **Google AI Studio API key** if you set `LLM_PROVIDER=google`  

---

## Getting started

### 1. Clone and configure the backend

```bash
cd backend
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY (or GOOGLE_API_KEY), NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, etc.
```

With **Docker Compose**, `QDRANT_URL` and `REDIS_URL` are overridden for the backend container. On your **host**, keep `QDRANT_URL=http://localhost:6333` and `REDIS_URL=redis://localhost:6379` when running Uvicorn outside Compose.

### 2. Run everything with Docker Compose

From the **repository root**:

```bash
docker compose up --build
```

- Frontend: http://localhost:3000  
- Backend API: http://localhost:8000  
- Qdrant dashboard: http://localhost:6333/dashboard  
- Redis: `localhost:6379`  

Ensure `backend/.env` exists (Compose loads it via `env_file`) and Neo4j credentials point to a live database.

### 3. Local development (without rebuilding images)

**Backend**

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start Redis and Qdrant via `docker compose up redis qdrant` from the repo root, then align `QDRANT_URL` / `REDIS_URL` in `.env`.

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (includes Redis reachability) |
| `/api/upload` | POST | Upload PDFs; returns `audit_id` and document metadata |
| `/api/audit/{id}/stream` | GET | SSE stream of agent events |
| `/api/audit/{id}/report` | GET | Completed audit report (JSON) |
| `/api/audit/{id}/graph` | GET | Knowledge graph (query params: `center_node`, `depth`, `view`) |
| `/api/audit/{id}/status` | GET | Session status and document list |
| `/api/documents/{doc_id}` | GET | Single document metadata |

### SSE event shapes

Events are JSON objects in SSE `data:` lines. Common `type` values:

- `connected` — session acknowledged; includes `audit_id`, `document_count`  
- `reasoning_step` — agent thought / tool call / tool result / finding  
- `report_ready` — report saved; includes `overall_risk`, `finding_count`  
- `complete` — audit pipeline finished  
- `error` — failure message  

---

## Configuration (backend)

Settings are loaded from `backend/.env` and validated by `app/config.py` (Pydantic Settings). Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `openrouter` | `openrouter` or `google` — selects the LLM backend |
| `OPENROUTER_API_KEY` | — | Required when `LLM_PROVIDER=openrouter` |
| `OPENROUTER_MODEL` | `openai/gpt-oss-20b:nitro` | Planner, report writer, document classification, graph-entity extraction |
| `OPENROUTER_MODEL_NER_ARABIC` | `qwen/qwen3-32b` | Entity / NER extraction per chunk — never substituted by `OPENROUTER_MODEL` |
| `OPENROUTER_MODEL_RELATION` | `anthropic/claude-sonnet-4.6` | Cross-checker contradiction adjudication — never substituted by `OPENROUTER_MODEL` |
| `OPENROUTER_REASONING` | `true` | Enables reasoning payload for `OPENROUTER_MODEL` only |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=google` |
| `GOOGLE_MODEL` | `gemini-2.5-pro` | Used for all three roles when `LLM_PROVIDER=google` |
| `NEO4J_URI` | `bolt://localhost:7687` | Graph database URI |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | — | |
| `NEO4J_DATABASE` | `neo4j` | |
| `QDRANT_URL` | `http://localhost:6333` | |
| `QDRANT_API_KEY` | — | Required for Qdrant Cloud |
| `REDIS_URL` | `redis://localhost:6379` | Session and report storage (7-day TTL) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Multilingual sentence embedding model |
| `EXTRACTION_CONCURRENCY` | `3` | Parallel LLM calls inside the entity extractor; reduce to `1` on free-tier rate limits |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins |

Frontend API base URL: `NEXT_PUBLIC_API_URL` in `frontend/.env.local` (defaults to `http://localhost:8000`).

---

## Repository layout

```
AuditMind/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, SSE routes, thread pool
│   │   ├── config.py               # Pydantic Settings (OpenRouter, Google, Neo4j, Qdrant, Redis)
│   │   ├── agents/
│   │   │   ├── graph.py            # LangGraph StateGraph pipeline wiring
│   │   │   ├── extraction.py       # Concurrent entity extraction → Neo4j
│   │   │   ├── planner.py          # Audit checklist generation
│   │   │   ├── cross_checker.py    # 4-phase contradiction detection engine
│   │   │   └── report_writer.py    # Structured report synthesis
│   │   ├── tools/
│   │   │   ├── hybrid_retriever.py # Rule-based router: Qdrant + Neo4j merged results
│   │   │   ├── neo4j_tools.py      # Graph search + contradiction detection
│   │   │   ├── planner_tools.py    # Checklist generation LLM tool
│   │   │   └── web_search.py       # DuckDuckGo entity verification
│   │   ├── services/
│   │   │   ├── document_processor.py  # PDF ingestion, OCR, chunking, Qdrant storage
│   │   │   ├── entity_extractor.py    # Async LLM entity/relationship extraction
│   │   │   ├── vector_store.py        # Qdrant client wrapper
│   │   │   ├── graph_builder.py       # Neo4j CRUD + schema init
│   │   │   ├── redis_store.py         # Async session/report persistence
│   │   │   └── reconciliation_payload.py  # Contract/invoice/bank reconciliation snapshot
│   │   ├── models/
│   │   │   ├── schemas.py          # AuditReport, Finding, ChecklistItem, DocumentMeta, Entity
│   │   │   └── state.py            # AuditState (LangGraph TypedDict)
│   │   └── utils/
│   │       ├── llm_factory.py      # get_llm() — OpenRouter / Google selector
│   │       ├── money_parse.py      # Deterministic monetary amount parsing
│   │       ├── arabic_normalizer.py
│   │       ├── canonical_id.py     # Stable hash IDs for dedup
│   │       └── text_amount_scan.py
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Home: file upload + language/model config
│   │   │   └── audit/[id]/         # Audit workspace: findings, reconciliation, report, reasoning
│   │   ├── components/
│   │   │   ├── FileUpload.tsx
│   │   │   ├── ReasoningPanel.tsx  # Live agent step feed
│   │   │   ├── FindingsTable.tsx
│   │   │   ├── FindingDetailSheet.tsx
│   │   │   ├── AuditReport.tsx     # Full report view, RTL-aware, JSON export
│   │   │   └── ReconciliationPanel.tsx  # Contract / invoice / bank variance card
│   │   └── hooks/                  # useAuditStream (SSE client)
│   ├── Dockerfile
│   └── package.json
└── docker-compose.yml
```

---

## One-line pitch (resume / portfolio)

> Bilingual agentic financial auditor: LangGraph + OpenRouter/Gemini, hybrid RAG (Qdrant + Neo4j), EasyOCR for Arabic/English PDFs, FastAPI SSE, Next.js reasoning UI with reconciliation panel.
