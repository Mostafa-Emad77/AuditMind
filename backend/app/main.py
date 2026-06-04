"""AuditMind FastAPI application — main entry point."""
import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models.schemas import (
    AuditReport,
    AuditSession,
    GraphData,
    UploadResponse,
)
from app.services.document_processor import ingest_document
from app.services.graph_builder import close_driver, get_entity_graph
from app.services.redis_store import (
    close_redis,
    find_document,
    get_redis,
    get_report,
    get_session,
    save_report,
    save_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound work (OCR, embedding) so it doesn't block the event loop.
# OCR is the bottleneck on multi-doc uploads; size to the core count so concurrent
# ingestion (asyncio.gather below) actually utilizes the machine.
_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AuditMind API starting up...")
    # Warm up Redis connection so first request is fast
    try:
        await get_redis()
    except Exception as e:
        logger.warning("Redis connection failed on startup: %s — will retry on first request.", e)
    yield
    logger.info("AuditMind API shutting down...")
    await close_redis()
    close_driver()


settings = get_settings()

app = FastAPI(
    title="AuditMind API",
    description="Bilingual Agentic Financial Document Auditor (Arabic/English)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    redis_ok = False
    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "version": "1.0.0",
        "llm_model": settings.openrouter_model,
        "redis": "connected" if redis_ok else "unavailable",
    }


# ─── Document Upload ──────────────────────────────────────────────────────────

@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    report_language: str = Query(default="english", regex="^(arabic|english)$"),
):
    """
    Upload one or more PDF documents to start an audit session.
    Heavy work (OCR + embedding) runs in a thread pool so the API stays responsive.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    audit_id = str(uuid.uuid4())

    # Read file bytes while we're still in the async context
    file_payloads: list[tuple[str, bytes]] = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are supported.",
            )
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is empty.")
        file_payloads.append((file.filename, content))

    # Run CPU-heavy ingestion (OCR + embedding) concurrently in the thread pool.
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(_executor, ingest_document, content, filename)
        for filename, content in file_payloads
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    document_metas = []
    for (filename, _content), result in zip(file_payloads, results):
        if isinstance(result, Exception):
            logger.error("Failed to ingest '%s': %s", filename, result)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process '{filename}': {str(result)}",
            )
        document_metas.append(result)
        logger.info("Ingested document '%s' for audit %s", filename, audit_id)

    session = AuditSession(
        audit_id=audit_id,
        documents=document_metas,
        status="pending",
        report_language=report_language,
    )
    await save_session(session)

    return UploadResponse(
        audit_id=audit_id,
        documents=document_metas,
        message=f"Successfully uploaded {len(document_metas)} document(s). Ready to audit.",
    )


# ─── Audit Streaming ──────────────────────────────────────────────────────────

@app.get("/api/audit/{audit_id}/stream")
async def stream_audit(audit_id: str):
    """
    Start the audit and stream agent reasoning steps as Server-Sent Events.

    Event types:
    - connected       — session acknowledged
    - reasoning_step  — agent thought / tool call / tool result / finding
    - report_ready    — report has been generated and saved
    - complete        — audit pipeline finished
    - error           — an error occurred
    """
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Audit session '{audit_id}' not found.")

    if session.status == "completed":
        raise HTTPException(
            status_code=409,
            detail="This audit session has already completed. Retrieve the report instead.",
        )

    if session.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Audit is already running. Connect to the existing stream or wait for it to complete.",
        )

    # Mark as processing and persist immediately
    session.status = "processing"
    await save_session(session)

    async def event_generator() -> AsyncIterator[str]:
        from app.agents.graph import audit_graph
        from app.models.state import AuditState

        # Re-fetch session inside generator to get latest state
        current_session = await get_session(audit_id)
        if not current_session:
            yield f"data: {json.dumps({'type': 'error', 'audit_id': audit_id, 'message': 'Session expired'})}\n\n"
            return

        initial_state: AuditState = {
            "messages": [],
            "audit_id": audit_id,
            "documents": current_session.documents,
            "checklist": [],
            "findings": [],
            "report": None,
            "reasoning_trace": [],
            "report_language": current_session.report_language,
            "extraction_complete": False,
            "error": None,
        }

        yield f"data: {json.dumps({'type': 'connected', 'audit_id': audit_id, 'document_count': len(current_session.documents)})}\n\n"

        try:
            async for event in audit_graph.astream(
                initial_state,
                stream_mode=["custom", "updates"],
            ):
                if isinstance(event, tuple):
                    mode, data = event
                    if mode == "custom":
                        yield f"data: {json.dumps(data, default=str)}\n\n"
                    elif mode == "updates":
                        for _node_name, node_update in data.items():
                            if node_update.get("report"):
                                report: AuditReport = node_update["report"]
                                await save_report(report)
                                yield f"data: {json.dumps({'type': 'report_ready', 'audit_id': audit_id, 'overall_risk': report.overall_risk, 'finding_count': len(report.findings)}, default=str)}\n\n"
                elif isinstance(event, dict):
                    if event.get("report"):
                        report = event["report"]
                        await save_report(report)
                        yield f"data: {json.dumps({'type': 'report_ready', 'audit_id': audit_id, 'overall_risk': report.overall_risk, 'finding_count': len(report.findings)}, default=str)}\n\n"

            # Mark completed and persist
            current_session.status = "completed"
            current_session.completed_at = datetime.utcnow()
            await save_session(current_session)

            yield f"data: {json.dumps({'type': 'complete', 'audit_id': audit_id, 'status': 'completed'})}\n\n"

        except Exception as e:
            logger.error("Audit stream error for %s: %s", audit_id, e)
            current_session.status = "failed"
            current_session.error = str(e)
            await save_session(current_session)
            yield f"data: {json.dumps({'type': 'error', 'audit_id': audit_id, 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Report Retrieval ─────────────────────────────────────────────────────────

@app.get("/api/audit/{audit_id}/report")
async def get_audit_report(audit_id: str):
    """Retrieve the completed audit report."""
    report = await get_report(audit_id)
    if not report:
        session = await get_session(audit_id)
        if not session:
            raise HTTPException(status_code=404, detail="Audit session not found.")
        if session.status != "completed":
            raise HTTPException(
                status_code=202,
                detail=f"Audit is still {session.status}. Poll this endpoint or connect to the stream.",
            )
        raise HTTPException(status_code=404, detail="Report not found despite completed status.")
    return report.model_dump(mode="json")


# ─── Knowledge Graph ──────────────────────────────────────────────────────────

@app.get("/api/audit/{audit_id}/graph", response_model=GraphData)
async def get_graph(
    audit_id: str,
    center_node: str | None = Query(default=None),
    depth: int = Query(default=1, ge=1, le=3),
    view: str = Query(default="canonical", pattern="^(canonical|raw)$"),
):
    """Retrieve the knowledge graph (nodes + edges) for visualization."""
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")

    doc_ids = [d.doc_id for d in session.documents]
    try:
        return get_entity_graph(doc_ids, center_node=center_node, depth=depth, view=view)
    except Exception as e:
        logger.error("Failed to retrieve graph for audit %s: %s", audit_id, e)
        raise HTTPException(status_code=500, detail=f"Graph retrieval failed: {str(e)}")


# ─── Session Status ───────────────────────────────────────────────────────────

@app.get("/api/audit/{audit_id}/status")
async def get_status(audit_id: str):
    """Get the current status of an audit session."""
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return {
        "audit_id": audit_id,
        "status": session.status,
        "document_count": len(session.documents),
        "documents": [
            {"doc_id": d.doc_id, "filename": d.filename, "doc_type": d.doc_type}
            for d in session.documents
        ],
        "created_at": session.created_at.isoformat(),
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }


# ─── Document Details ─────────────────────────────────────────────────────────

@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Retrieve metadata for a specific document."""
    doc = await find_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc.model_dump()
