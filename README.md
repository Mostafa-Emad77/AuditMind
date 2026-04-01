# AuditMind

**Bilingual Agentic Financial Document Auditor for Arabic/English Enterprises**

AuditMind is an AI-powered auditing platform that autonomously analyzes financial documents in Arabic and English, detects cross-document contradictions using a knowledge graph, and streams its full reasoning process in real-time. Built for Egyptian and MENA enterprises dealing with mixed-language financial documents.

---

## What It Does

Upload invoices, contracts, bank statements, and balance sheets (scanned or digital, Arabic or English). AuditMind:

1. **Extracts** text via ArabicOCR + Tesseract (scanned PDFs) or PyMuPDF (digital)
2. **Builds a knowledge graph** of entities and relationships (Neo4j) + semantic chunks (Qdrant)
3. **Plans the audit** — generates a targeted checklist based on document types
4. **Cross-checks** across all documents using hybrid RAG (vector + graph traversal)
5. **Detects contradictions** — e.g., Invoice says 50,000 EGP, Contract says 45,000 EGP
6. **Produces a report** in Arabic or English with source citations
7. **Streams all reasoning** live to a UI panel — like watching a senior auditor think out loud

---

## Architecture

```
Documents (PDF/scans)
        ↓
[Extraction Agent]           ArabicOCR + Tesseract + PyMuPDF
  → Qdrant (chunks)          Semantic vector search
  → Neo4j (knowledge graph)  Entity relationship traversal
        ↓
[Audit Planner Agent]        Generates custom checklist per doc combo
        ↓
[Cross-Checker Agent]        Hybrid RAG: Qdrant + Neo4j graph traversal
  → compare_values()         Contradiction detection with severity scoring
  → search_web()             External verification (company names, rates)
        ↓
[Report Writer Agent]        Structured Arabic/English audit report
        ↓
[FastAPI + SSE]              Streams reasoning steps to React UI
        ↓
[Next.js Frontend]           Live reasoning panel + knowledge graph viz
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph (multi-agent StateGraph) |
| LLM | Claude / GPT-4o / Gemini / Groq (configurable) |
| OCR | ArabicOCR + Tesseract (Arabic + English) |
| Vector DB | Qdrant |
| Graph DB | Neo4j Aura (free tier) |
| Backend | FastAPI + SSE streaming |
| Frontend | Next.js 15 + Tailwind CSS + Framer Motion |
| Containerization | Docker + Docker Compose |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- Tesseract OCR with Arabic language pack
- Neo4j Aura account (free tier): https://neo4j.com/cloud/platform/aura-graph-database/

### 1. Clone and configure

```bash
cd backend
cp .env.example .env
# Edit .env with your API keys
```

Required `.env` values:

```env
LLM_PROVIDER=anthropic          # or openai, google, groq
ANTHROPIC_API_KEY=sk-ant-...

NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

QDRANT_URL=http://localhost:6333
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Qdrant dashboard: http://localhost:6333/dashboard

### 3. Run locally (development)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload PDFs, returns `audit_id` |
| `/api/audit/{id}/stream` | GET | SSE stream of agent reasoning events |
| `/api/audit/{id}/report` | GET | Completed audit report (JSON) |
| `/api/audit/{id}/graph` | GET | Knowledge graph nodes + edges |
| `/api/audit/{id}/status` | GET | Current audit status |
| `/health` | GET | Health check |

### SSE Event Types

```typescript
type SSEEvent =
  | { type: "connected"; audit_id: string }
  | { type: "reasoning_step"; step: ReasoningStep }
  | { type: "report_ready"; overall_risk: string; finding_count: number }
  | { type: "complete"; audit_id: string }
  | { type: "error"; message: string }
```

---

## LLM Provider Configuration

Switch providers with zero code changes via environment variable:

```env
LLM_PROVIDER=anthropic   # claude-3-5-sonnet-20241022
LLM_PROVIDER=openai      # gpt-4o
LLM_PROVIDER=google      # gemini-2.0-flash
LLM_PROVIDER=groq        # llama-3.3-70b-versatile
```

Override specific model:
```env
LLM_MODEL=claude-3-opus-20240229
```

---

## Project Structure

```
AuditMind/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + SSE endpoints
│   │   ├── config.py            # Pydantic Settings
│   │   ├── agents/
│   │   │   ├── graph.py         # Master LangGraph StateGraph
│   │   │   ├── extraction.py    # Extraction Agent
│   │   │   ├── planner.py       # Audit Planner Agent
│   │   │   ├── cross_checker.py # Cross-Checker Agent (core)
│   │   │   └── report_writer.py # Report Writer Agent
│   │   ├── tools/               # @tool decorated functions
│   │   │   ├── hybrid_retriever.py  # Qdrant + Neo4j combined
│   │   │   ├── financial.py         # Amount extraction & comparison
│   │   │   ├── neo4j_tools.py       # Graph traversal tools
│   │   │   └── web_search.py        # External verification
│   │   ├── services/
│   │   │   ├── document_processor.py  # PDF → OCR → chunks
│   │   │   ├── entity_extractor.py    # LLM entity extraction
│   │   │   ├── graph_builder.py       # Neo4j graph operations
│   │   │   └── vector_store.py        # Qdrant operations
│   │   └── utils/
│   │       ├── arabic_normalizer.py   # Arabic text normalization
│   │       └── llm_factory.py         # Multi-provider LLM factory
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx              # Upload / landing page
│       │   └── audit/[id]/page.tsx   # Audit session page
│       ├── components/
│       │   ├── ReasoningPanel.tsx    # Live agent reasoning UI
│       │   ├── FindingsTable.tsx     # Severity-sorted findings
│       │   ├── AuditReport.tsx       # Final report display
│       │   ├── KnowledgeGraph.tsx    # Interactive graph canvas
│       │   └── FileUpload.tsx        # Drag-and-drop uploader
│       └── hooks/
│           └── useAuditStream.ts     # SSE consumer hook
└── docker-compose.yml
```

---

## CV Summary

> "Built AuditMind, a bilingual agentic financial auditor using LangGraph and Claude API. Implemented cross-document contradiction detection across Arabic/English PDFs using Qdrant vector search + Neo4j GraphRAG, ArabicOCR, and a live chain-of-thought reasoning UI built in Next.js. The system autonomously detects amount mismatches, party inconsistencies, and date conflicts across mixed-language document sets."
