"""AuditMind FastAPI application — main entry point."""
import asyncio
import functools
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.dependencies import require_api_key
from app.models.schemas import (
    AuditListResponse,
    AuditReport,
    AuditSession,
    AuditSummary,
    ChatRequest,
    GraphData,
    TriageRecord,
    TriageRequest,
    UploadFileError,
    UploadResponse,
)
from app.services.document_processor import ingest_document
from app.services.graph_builder import close_driver, delete_audit_documents, get_entity_graph, init_graph_schema
from app.services.vector_store import delete_docs_chunks, init_collection
from app.services.redis_store import (
    add_suppressed_signature,
    append_audit_event,
    append_chat_messages,
    clear_audit_events,
    close_redis,
    find_document,
    get_audit_events,
    get_chat_history,
    get_redis,
    get_report,
    get_session,
    get_triage,
    is_heartbeat_alive,
    list_sessions,
    purge_audit,
    remove_suppressed_signature,
    save_report,
    save_session,
    set_triage,
    touch_heartbeat,
)
from app.utils.finding_signature import finding_signature

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound work (OCR, embedding) so it doesn't block the event loop.
# OCR is the bottleneck on multi-doc uploads; size to the core count so concurrent
# ingestion (asyncio.gather below) actually utilizes the machine.
_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# In-process registry of running audit pipeline tasks, keyed by audit_id.
# The pipeline itself runs as a free-standing asyncio.Task (not tied to any SSE
# connection), so a client disconnecting from /stream no longer cancels the audit.
# This dict only prevents duplicate task creation within a single process/worker;
# cross-process/restart staleness is handled by the heartbeat TTL (see below).
_running_audits: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AuditMind API starting up...")
    if not get_settings().api_keys_list:
        logger.warning(
            "API_KEYS is not set — all /api/* routes are unauthenticated. "
            "Set API_KEYS (comma-separated) to enable access control."
        )
    # Warm up Redis connection so first request is fast
    try:
        await get_redis()
    except Exception as e:
        logger.warning("Redis connection failed on startup: %s — will retry on first request.", e)
    # Initialize the Neo4j graph schema once at boot (was previously re-run per audit).
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, init_graph_schema)
        logger.info("Neo4j schema initialized.")
    except Exception as e:
        logger.warning("Neo4j schema init failed on startup (non-fatal): %s", e)
    # Create the Qdrant collection + payload indexes once at boot (was re-run per document).
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, init_collection)
        logger.info("Qdrant collection initialized.")
    except Exception as e:
        logger.warning("Qdrant collection init failed on startup (non-fatal): %s", e)
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

# All /api/* routes require a valid API key when API_KEYS is configured; /health stays open.
api_router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


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

@api_router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    report_language: str = Query(default="english", regex="^(arabic|english)$"),
    api_key: str = Depends(require_api_key),
):
    """
    Upload one or more PDF documents to start an audit session.
    Heavy work (OCR + embedding) runs in a thread pool so the API stays responsive.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=422,
            detail=f"Too many files: {len(files)} uploaded, max is {settings.max_upload_files}.",
        )

    audit_id = str(uuid.uuid4())
    max_bytes = settings.max_upload_file_size_mb * 1024 * 1024

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
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' exceeds the {settings.max_upload_file_size_mb} MB limit.",
            )
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=422,
                detail=f"File '{file.filename}' is not a valid PDF (bad file signature).",
            )
        try:
            probe = fitz.open(stream=content, filetype="pdf")
            page_count = len(probe)
            probe.close()
        except Exception:
            raise HTTPException(
                status_code=422,
                detail=f"File '{file.filename}' could not be parsed as a PDF.",
            )
        if page_count > settings.max_upload_pages:
            raise HTTPException(
                status_code=422,
                detail=f"File '{file.filename}' has {page_count} pages, max is {settings.max_upload_pages}.",
            )
        file_payloads.append((file.filename, content))

    # Run CPU-heavy ingestion (OCR + embedding) concurrently in the thread pool.
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(_executor, ingest_document, content, filename)
        for filename, content in file_payloads
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    document_metas = []
    failed_files: list[UploadFileError] = []
    for (filename, _content), result in zip(file_payloads, results):
        if isinstance(result, Exception):
            logger.error("Failed to ingest '%s': %s", filename, result)
            failed_files.append(UploadFileError(filename=filename, error=str(result)))
            continue
        document_metas.append(result)
        logger.info("Ingested document '%s' for audit %s", filename, audit_id)

    if not document_metas:
        raise HTTPException(
            status_code=422,
            detail=f"All {len(failed_files)} file(s) failed to process: "
            + "; ".join(f"{f.filename} ({f.error})" for f in failed_files),
        )

    session = AuditSession(
        audit_id=audit_id,
        documents=document_metas,
        status="pending",
        report_language=report_language,
        api_key=api_key,
    )
    await save_session(session)

    message = f"Successfully uploaded {len(document_metas)} document(s). Ready to audit."
    if failed_files:
        message += f" {len(failed_files)} file(s) failed and were skipped."

    return UploadResponse(
        audit_id=audit_id,
        documents=document_metas,
        message=message,
        failed=failed_files,
    )


# ─── Audit Execution (background task, decoupled from any SSE connection) ────

async def _run_audit_pipeline(audit_id: str) -> None:
    """
    Run the full LangGraph audit pipeline to completion, persisting every SSE-shaped
    event to Redis as it goes (see `append_audit_event`). This is a free-standing
    asyncio.Task — it is NOT tied to any client connection, so a client closing its
    `/stream` connection has no effect on the audit; it keeps running and can be
    resumed by any later `/stream` subscriber.
    """
    from app.agents.graph import audit_graph
    from app.models.state import AuditState

    current_session = await get_session(audit_id)
    if not current_session:
        await append_audit_event(audit_id, {"type": "error", "audit_id": audit_id, "message": "Session expired"})
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
        "api_key": current_session.api_key,
    }

    await append_audit_event(
        audit_id,
        {"type": "connected", "audit_id": audit_id, "document_count": len(current_session.documents)},
    )

    try:
        async for event in audit_graph.astream(initial_state, stream_mode=["custom", "updates"]):
            await touch_heartbeat(audit_id)
            if isinstance(event, tuple):
                mode, data = event
                if mode == "custom":
                    await append_audit_event(audit_id, data)
                elif mode == "updates":
                    for _node_name, node_update in data.items():
                        if node_update.get("report"):
                            report: AuditReport = node_update["report"]
                            await save_report(report)
                            await append_audit_event(audit_id, {
                                "type": "report_ready",
                                "audit_id": audit_id,
                                "overall_risk": report.overall_risk,
                                "finding_count": len(report.findings),
                            })

        current_session.status = "completed"
        current_session.completed_at = datetime.utcnow()
        await save_session(current_session)
        await append_audit_event(audit_id, {"type": "complete", "audit_id": audit_id, "status": "completed"})

    except Exception as e:
        logger.error("Audit pipeline error for %s: %s", audit_id, e)
        current_session.status = "failed"
        current_session.error = str(e)
        await save_session(current_session)
        await append_audit_event(audit_id, {"type": "error", "audit_id": audit_id, "message": str(e)})
    finally:
        _running_audits.pop(audit_id, None)


async def _ensure_audit_running(audit_id: str, session: AuditSession) -> AuditSession:
    """
    Idempotently (re)start the background pipeline task for an audit.

    - completed: no-op, caller should fetch the report.
    - processing + alive heartbeat: already running (in this or another worker), no-op.
    - processing + stale heartbeat, pending, or failed: (re)start from scratch.
    """
    if session.status == "completed":
        return session

    if session.status == "processing":
        if await is_heartbeat_alive(audit_id) or audit_id in _running_audits:
            return session
        logger.warning("Recovering stale 'processing' session %s (heartbeat expired) — restarting.", audit_id)

    await clear_audit_events(audit_id)
    session.status = "processing"
    session.error = None
    await save_session(session)
    await touch_heartbeat(audit_id)

    task = asyncio.create_task(_run_audit_pipeline(audit_id))
    _running_audits[audit_id] = task
    return session


@api_router.post("/audit/{audit_id}/start")
async def start_audit(audit_id: str):
    """Start (or resume) an audit's background pipeline. Idempotent — safe to call repeatedly."""
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Audit session '{audit_id}' not found.")
    session = await _ensure_audit_running(audit_id, session)
    return {"audit_id": audit_id, "status": session.status}


@api_router.get("/audit/{audit_id}/stream")
async def stream_audit(audit_id: str):
    """
    Subscribe to an audit's Server-Sent Events. Starts the pipeline if it hasn't
    been started yet; otherwise replays every event persisted so far and then tails
    new ones as they're produced by the (independently running) background task.

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

    session = await _ensure_audit_running(audit_id, session)

    async def event_generator() -> AsyncIterator[str]:
        next_index = 0
        terminal_types = {"complete", "error"}
        while True:
            events = await get_audit_events(audit_id, start=next_index)
            for event in events:
                yield f"data: {json.dumps(event, default=str)}\n\n"
                next_index += 1
                if event.get("type") in terminal_types:
                    return
            if events:
                continue
            # No new events this pass — check if the run has finished without
            # emitting an explicit terminal event (defensive; normally it does).
            current = await get_session(audit_id)
            if not current:
                yield f"data: {json.dumps({'type': 'error', 'audit_id': audit_id, 'message': 'Session expired'})}\n\n"
                return
            if current.status == "completed":
                yield f"data: {json.dumps({'type': 'complete', 'audit_id': audit_id, 'status': 'completed'})}\n\n"
                return
            if current.status == "failed":
                yield f"data: {json.dumps({'type': 'error', 'audit_id': audit_id, 'message': current.error or 'Audit failed'})}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Report Retrieval ─────────────────────────────────────────────────────────

@api_router.get("/audit/{audit_id}/report")
async def get_audit_report(audit_id: str):
    """Retrieve the completed audit report.

    Returns 200 with `{"status": "processing"|"pending"|"failed", ...}` while the
    audit hasn't finished yet — the audit is still processing, not an error
    condition, so this intentionally avoids the anti-pattern of raising an
    HTTPException with a 2xx status code.
    """
    report = await get_report(audit_id)
    if not report:
        session = await get_session(audit_id)
        if not session:
            raise HTTPException(status_code=404, detail="Audit session not found.")
        if session.status != "completed":
            return {
                "audit_id": audit_id,
                "status": session.status,
                "message": f"Audit is still {session.status}. Poll this endpoint or connect to the stream.",
            }
        raise HTTPException(status_code=404, detail="Report not found despite completed status.")
    return report.model_dump(mode="json")


# ─── Conversational Q&A ───────────────────────────────────────────────────────

@api_router.get("/audit/{audit_id}/chat")
async def get_chat(audit_id: str):
    """Return the stored chat history for an audit (for rehydrating the UI)."""
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return {"audit_id": audit_id, "messages": await get_chat_history(audit_id)}


@api_router.post("/audit/{audit_id}/chat")
async def chat_with_audit(audit_id: str, body: ChatRequest):
    """
    Ask a question about the audited documents. Streams a retrieval-grounded answer
    as Server-Sent Events: `sources` (retrieved passages), `delta` (answer chunks),
    `done`, or `error`.
    """
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")

    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="A non-empty message is required.")

    report = await get_report(audit_id)  # may be None if the audit hasn't finished
    history = await get_chat_history(audit_id)

    async def event_generator() -> AsyncIterator[str]:
        from app.agents.qa_agent import answer_audit_question

        try:
            async for evt in answer_audit_question(
                audit_id=audit_id,
                session=session,
                report=report,
                message=message,
                history=history,
            ):
                yield f"data: {json.dumps(evt, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            logger.error("Chat stream error for %s: %s", audit_id, e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Finding triage ───────────────────────────────────────────────────────────

@api_router.get("/audit/{audit_id}/triage")
async def get_audit_triage(audit_id: str):
    """Return the triage map (finding_id → record) for an audit."""
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return {"audit_id": audit_id, "triage": await get_triage(audit_id)}


@api_router.post("/audit/{audit_id}/findings/{finding_id}/triage")
async def triage_finding(audit_id: str, finding_id: str, body: TriageRequest):
    """
    Record a reviewer's verdict on a finding (accepted / dismissed / false_positive).

    Marking a finding as a false positive stores its signature in a set scoped to the
    audit's API key so the cross-checker can suppress matching findings on future runs
    by that same caller. Changing the verdict away from false_positive lifts that
    suppression.
    """
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")

    report = await get_report(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="No report found for this audit.")

    finding = next((f for f in report.findings if f.finding_id == finding_id), None)
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found in this audit's report.")

    signature = finding_signature(finding)
    record = TriageRecord(status=body.status, note=body.note, signature=signature)
    triage_map = await set_triage(audit_id, finding_id, record.model_dump(mode="json"))

    if body.status == "false_positive":
        await add_suppressed_signature(session.api_key, signature)
    else:
        # Reviewer changed their mind — stop suppressing this signature.
        await remove_suppressed_signature(session.api_key, signature)

    return {"audit_id": audit_id, "finding_id": finding_id, "triage": triage_map[finding_id]}


# ─── Knowledge Graph ──────────────────────────────────────────────────────────

@api_router.get("/audit/{audit_id}/graph", response_model=GraphData)
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
        # get_entity_graph uses the sync Neo4j driver — offload so it doesn't block the event loop.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                get_entity_graph, doc_ids, center_node=center_node, depth=depth, view=view
            ),
        )
    except Exception as e:
        logger.error("Failed to retrieve graph for audit %s: %s", audit_id, e)
        raise HTTPException(status_code=500, detail=f"Graph retrieval failed: {str(e)}")


# ─── Audit Deletion (data lifecycle) ──────────────────────────────────────────

@api_router.delete("/audit/{audit_id}")
async def delete_audit(audit_id: str):
    """
    Permanently delete an audit: its Redis session/report/chat/triage/events, its
    Qdrant document chunks, and its Neo4j Document nodes (orphaned Entity nodes are
    pruned too — entities still referenced by other audits' documents are kept).
    """
    session = await get_session(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")

    doc_ids = [d.doc_id for d in session.documents]
    loop = asyncio.get_running_loop()

    if doc_ids:
        try:
            await loop.run_in_executor(_executor, delete_docs_chunks, doc_ids)
        except Exception as e:
            logger.warning("Failed to delete Qdrant chunks for audit %s: %s", audit_id, e)
        try:
            await loop.run_in_executor(_executor, delete_audit_documents, doc_ids)
        except Exception as e:
            logger.warning("Failed to delete Neo4j data for audit %s: %s", audit_id, e)

    await purge_audit(audit_id)
    _running_audits.pop(audit_id, None)

    return {"audit_id": audit_id, "deleted": True}


# ─── Audit Archive ─────────────────────────────────────────────────────────────

@api_router.get("/audits", response_model=AuditListResponse)
async def list_audits(api_key: str = Depends(require_api_key)):
    """List past audits for the caller's scope, newest first — powers the Archive page."""
    sessions = await list_sessions(api_key)
    summaries: list[AuditSummary] = []
    for session in sessions:
        overall_risk = None
        critical_count = 0
        warning_count = 0
        if session.status == "completed":
            report = await get_report(session.audit_id)
            if report:
                overall_risk = report.overall_risk
                critical_count = sum(1 for f in report.findings if f.severity == "critical")
                warning_count = sum(1 for f in report.findings if f.severity == "warning")
        summaries.append(AuditSummary(
            audit_id=session.audit_id,
            status=session.status,
            document_count=len(session.documents),
            filenames=[d.filename for d in session.documents],
            created_at=session.created_at,
            completed_at=session.completed_at,
            error=session.error,
            overall_risk=overall_risk,
            critical_count=critical_count,
            warning_count=warning_count,
        ))
    return AuditListResponse(audits=summaries)


# ─── Session Status ───────────────────────────────────────────────────────────

@api_router.get("/audit/{audit_id}/status")
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
        "error": session.error,
    }


# ─── Document Details ─────────────────────────────────────────────────────────

@api_router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Retrieve metadata for a specific document."""
    doc = await find_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc.model_dump()


app.include_router(api_router)
