from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid


class DocumentMeta(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    doc_type: Literal["invoice", "contract", "balance_sheet", "bank_statement", "audit_report", "unknown"] = "unknown"
    language: Literal["arabic", "english", "mixed", "unknown"] = "unknown"
    page_count: int = 0
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    dates_found: list[str] = Field(default_factory=list)
    amounts_found: list[str] = Field(default_factory=list)
    ocr_used: bool = False


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    normalized_text: str
    page_num: int
    chunk_index: int
    language: str
    entities: list[dict] = Field(default_factory=list)


AMOUNT_ROLES = Literal[
    # ── Contract-side ────────────────────────────────────────────────────────
    "total_contract_value",
    "milestone_scheduled",
    "retainer",
    # ── Invoice-side ─────────────────────────────────────────────────────────
    "total_invoice",
    "invoice_subtotal",
    "invoice_line_item",
    # An invoice RESTATING what it claims the contract total is. Deliberately
    # distinct from total_contract_value — comparing the two is how restatement
    # errors are caught, so they must never collapse into one role.
    "invoice_referenced_contract_value",
    # ── Bank-statement-side ──────────────────────────────────────────────────
    "transaction_debit",
    "transaction_credit",
    # Statement-level summary figures — used to VALIDATE the summed transaction rows,
    # never added to them (adding both double-counts the statement).
    "statement_total_debits",
    "statement_total_credits",
    # Cumulative account state, NOT a transaction value. Must never be summed,
    # compared against invoice/contract totals, or used in "total paid" math.
    "running_balance",
    "opening_balance",
    "closing_balance",
    # ── Other ────────────────────────────────────────────────────────────────
    "vat_tax",
    "late_fee",
    # Legacy: pre-dates the debit/credit split. Treated as transaction_debit by
    # reconciliation so historical graph data keeps working.
    "single_payment",
    "unknown",
]

# Roles that represent cumulative account state or a pre-aggregated total rather than
# an individual transaction value. Summing any of these is always a bug — it either
# double-counts the whole account history (balances) or double-counts the rows
# themselves (statement totals).
NON_SUMMABLE_ROLES = frozenset({
    "running_balance",
    "opening_balance",
    "closing_balance",
    "statement_total_debits",
    "statement_total_credits",
})

# Roles that count as money actually leaving the account. `single_payment` is the
# pre-split legacy role and is treated as a debit for backwards compatibility.
BANK_OUTFLOW_ROLES = frozenset({"transaction_debit", "single_payment"})


class Entity(BaseModel):
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: Literal["person", "company", "amount", "date", "contract_id", "invoice_id", "clause", "other"]
    value: str
    normalized_value: str
    source_doc_id: str
    source_page: int
    confidence: float = 1.0
    amount_role: Optional[AMOUNT_ROLES] = None
    # Set by LLM extraction (see entity_extractor prompt); used for contradiction / graph context
    source_language: Literal["arabic", "english", "mixed", "unknown"] = "unknown"
    coreference_note: Optional[str] = None
    # Amount entities: when both digit scripts appear, LLM fills these (no regex in pipeline)
    western_numeral_form: Optional[str] = None
    arabic_indic_numeral_form: Optional[str] = None
    numeral_mismatch: Optional[bool] = None
    # Deterministic numeric parse for amount entities (set by entity_extractor; never by LLM).
    # When populated, downstream contradiction logic compares numerically with tolerance
    # instead of doing string-equality on `normalized_value`.
    amount_value: Optional[float] = None
    amount_currency: Optional[str] = None
    # Person / signatory entities — enables authority-mismatch findings (e.g. an
    # amendment signed by a CFO where the original was signed by the Chairman).
    role_title: Optional[str] = None
    signing_authority_level: Optional[
        Literal["chairman", "ceo", "cfo", "director", "manager", "other", "unknown"]
    ] = None
    document_signed: Optional[str] = None


class Relationship(BaseModel):
    rel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # "signed_by", "references", "payment_for", "amount_of", "dated", etc.
    source_doc_id: str
    source_page: int
    confidence: float = 1.0


class ChecklistItem(BaseModel):
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    check_type: Literal["amount_match", "date_consistency", "party_match", "clause_completeness", "signature_check", "cross_doc_consistency", "other"]
    doc_ids_involved: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"


class Finding(BaseModel):
    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    checklist_item_id: Optional[str] = None
    severity: Literal["critical", "warning", "ok"]
    title: str
    description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_doc_id: Optional[str] = None
    source_page: Optional[int] = None
    conflicting_doc_id: Optional[str] = None
    conflicting_page: Optional[int] = None
    evidence: list[str] = Field(default_factory=list)
    recommendation: Optional[str] = None


class ReasoningStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: Literal["extraction", "planner", "cross_checker", "report_writer"]
    step_type: Literal["thought", "tool_call", "tool_result", "finding", "summary"]
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AuditSession(BaseModel):
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    documents: list[DocumentMeta] = Field(default_factory=list)
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    report_language: Literal["arabic", "english"] = "english"
    error: Optional[str] = None
    # Scope for the suppressed-findings feedback loop; "default" when API-key auth is disabled.
    api_key: str = "default"


class ReconciliationSnapshot(BaseModel):
    """Structured cross-document financial totals for the Reconciliation UI card."""

    currency: str = "EGP"
    contract_total: Optional[float] = None
    invoice_total: Optional[float] = None
    bank_paid_total: Optional[float] = None
    variance_vs_contract: Optional[float] = None
    notes: Optional[str] = None


class EntityConflictRow(BaseModel):
    """Single cross-document conflict row (not full entity extraction)."""

    entity_label: str
    doc_a_id: str
    doc_a_value: str
    doc_b_id: str
    doc_b_value: str
    severity: Literal["critical", "warning"]
    # Human-readable explanation of WHY these two values were compared, derived from
    # the matched role pair (e.g. "contract total vs invoice's stated contract
    # reference"). Free-form rather than an enum so the reason can be specific.
    conflict_type: str = "amount"
    anchor_hint: Optional[str] = None


class AuditReport(BaseModel):
    audit_id: str
    title: str
    language: Literal["arabic", "english"]
    executive_summary: str
    documents_reviewed: list[str]
    findings: list[Finding]
    recommendations: list[str]
    overall_risk: Literal["critical", "high", "medium", "low", "clean"]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    reconciliation_snapshot: Optional[ReconciliationSnapshot] = None
    entity_conflicts: list[EntityConflictRow] = Field(default_factory=list)


# ─── API Request / Response schemas ─────────────────────────────────────────

class UploadFileError(BaseModel):
    filename: str
    error: str


class UploadResponse(BaseModel):
    audit_id: str
    documents: list[DocumentMeta]
    message: str
    failed: list[UploadFileError] = Field(default_factory=list)


class AuditSummary(BaseModel):
    """Lightweight per-audit summary row for the Archive list."""

    audit_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    document_count: int
    filenames: list[str]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    overall_risk: Optional[Literal["critical", "high", "medium", "low", "clean"]] = None
    critical_count: int = 0
    warning_count: int = 0


class AuditListResponse(BaseModel):
    audits: list[AuditSummary]


TriageStatus = Literal["accepted", "dismissed", "false_positive"]


class TriageRequest(BaseModel):
    status: TriageStatus
    note: Optional[str] = None


class TriageRecord(BaseModel):
    status: TriageStatus
    note: Optional[str] = None
    signature: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    message: str


class GraphNode(BaseModel):
    id: str
    label: str
    node_type: str
    doc_id: Optional[str] = None
    mention_count: int = 1
    degree: int = 0
    is_center: bool = False
    raw_ids: list[str] = Field(default_factory=list)
    properties: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    is_contradiction: bool = False
    count: int = 1
    show_label: Optional[bool] = None
    properties: dict = Field(default_factory=dict)


class GraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center_node: Optional[str] = None
    view: str = "canonical"
