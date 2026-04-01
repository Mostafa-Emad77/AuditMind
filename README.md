# AuditMind

**Bilingual agentic financial document auditor for Arabic/English workflows**

AuditMind analyzes financial PDFs in Arabic and English, cross-checks them with hybrid RAG (vector + graph), and streams agent reasoning to the UI in real time. It targets Egyptian and MENA use cases with mixed-language documents.

---

## What it does

Upload invoices, contracts, bank statements, or balance sheets (scanned or digital). AuditMind:

1. **Extracts** text — PyMuPDF for digital PDFs; Tesseract (Arabic/English) and optional Arabic OCR for scans  
2. **Stores** semantic chunks in **Qdrant** and builds an entity graph in **Neo4j**  
3. **Plans** a targeted audit checklist from document types  
4. **Cross-checks** across documents via hybrid retrieval and graph traversal  
5. **Surfaces contradictions** (e.g. amount mismatches) with severity  
6. **Writes** a structured report in Arabic or English  
7. **Streams** reasoning steps over **SSE** to the Next.js UI  

---

## Architecture

```
PDFs
  → Extraction (OCR / PyMuPDF)
      → Qdrant (chunks)     +  Neo4j (entities & relations)
  → Planner → Cross-checker (hybrid RAG + tools)
      → Report writer
  → FastAPI (SSE) → Next.js UI
```

**Infra (Docker Compose):** Redis (sessions), Qdrant, FastAPI backend, Next.js frontend. **Neo4j** is external (e.g. [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/)) — configure `NEO4J_*` in `backend/.env`.

---

## Tech stack

| Layer | Technology |
|--------|------------|
| Agents | LangGraph (`StateGraph`) |
| LLM | [OpenRouter](https://openrouter.ai/) (any supported model slug) |
| OCR | Tesseract (ara/eng), PyMuPDF; optional Arabic OCR where supported |
| Vector DB | Qdrant |
| Graph DB | Neo4j (Aura or self-hosted) |
| Cache / sessions | Redis |
| Backend | FastAPI, SSE |
| Frontend | Next.js 15, React 19, Tailwind CSS, Framer Motion |
| Ops | Docker, Docker Compose |

---

## Prerequisites

- **Python 3.11+** (matches `backend/Dockerfile`)  
- **Node.js 20+**  
- **Docker** with Compose v2 (`docker compose`)  
- **Tesseract** with Arabic + English data (handled in the backend Docker image; install locally if you run the API on the host)  
- **Neo4j** reachable from the backend (Aura free tier is fine)  
- **OpenRouter** API key  

---

## Getting started

### 1. Clone and configure the backend

```bash
cd backend
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, etc.
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

Start Redis and Qdrant (e.g. `docker compose up redis qdrant` from the repo root, or your own instances) and align `QDRANT_URL` / `REDIS_URL` in `.env`.

**Frontend**

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (includes Redis reachability when configured) |
| `/api/upload` | POST | Upload PDFs; returns `audit_id` |
| `/api/audit/{id}/stream` | GET | SSE stream of agent events |
| `/api/audit/{id}/report` | GET | Completed audit report (JSON) |
| `/api/audit/{id}/graph` | GET | Knowledge graph (query: `center_node`, `depth`, `view`) |
| `/api/audit/{id}/status` | GET | Session status and document list |
| `/api/documents/{doc_id}` | GET | Single document metadata |

### SSE event shapes (simplified)

Events are JSON objects in SSE `data:` lines. Common `type` values:

- `connected` — includes `audit_id`, `document_count`  
- `reasoning_step` — includes `step` (agent/tool trace)  
- `report_ready` — includes `overall_risk`, `finding_count`  
- `complete` — audit finished  
- `error` — failure message  

---

## Configuration (backend)

Settings come from `backend/.env` and map to `app/config.py`. Important variables:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Required for LLM calls |
| `OPENROUTER_MODEL` | Model slug (default in code: `google/gemini-3-flash-preview`) |
| `OPENROUTER_REASONING` | `true` / `false` — reasoning payload for supported models |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Graph database |
| `QDRANT_URL`, `QDRANT_API_KEY` (if applicable) | Vector store |
| `REDIS_URL` | Session / report storage |
| `CORS_ORIGINS` | Comma-separated browser origins |

Frontend URL for API calls: `NEXT_PUBLIC_API_URL` in `frontend/.env.local` (defaults to `http://localhost:8000` in code).

---

## Repository layout

```
AuditMind/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI + SSE
│   │   ├── config.py               # Settings (OpenRouter, Neo4j, Qdrant, Redis)
│   │   ├── agents/
│   │   │   ├── graph.py            # LangGraph pipeline
│   │   │   ├── extraction.py
│   │   │   ├── planner.py
│   │   │   ├── cross_checker.py
│   │   │   └── report_writer.py
│   │   ├── tools/                  # hybrid_retriever, neo4j_tools, qdrant_tools, …
│   │   ├── services/               # document_processor, vector_store, graph_builder, redis_store, …
│   │   ├── models/
│   │   └── utils/
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app/                    # Next.js App Router
│   │   ├── components/             # ReasoningPanel, FindingsTable, AuditReport, ReconciliationPanel, …
│   │   └── hooks/                  # useAuditStream (SSE)
│   ├── Dockerfile
│   └── .env.local.example
└── docker-compose.yml
```

---

## One-line pitch (resume / portfolio)

> Bilingual agentic financial auditor: LangGraph + OpenRouter, hybrid RAG (Qdrant + Neo4j), OCR for Arabic/English PDFs, FastAPI SSE, Next.js reasoning UI.
