// ─── Document Types ──────────────────────────────────────────────────────────

export type DocType =
  | "invoice"
  | "contract"
  | "balance_sheet"
  | "bank_statement"
  | "audit_report"
  | "payment_certificate"
  | "qs_report"
  | "board_resolution"
  | "unknown";

export type Language = "arabic" | "english" | "mixed" | "unknown";

export interface DocumentMeta {
  doc_id: string;
  filename: string;
  doc_type: DocType;
  language: Language;
  page_count: number;
  upload_time: string;
  dates_found: string[];
  amounts_found: string[];
  ocr_used: boolean;
}

// ─── Audit Types ─────────────────────────────────────────────────────────────

export type Severity = "critical" | "warning" | "ok";
export type OverallRisk = "critical" | "high" | "medium" | "low" | "clean";
export type AuditStatus = "pending" | "processing" | "completed" | "failed";

export interface Finding {
  finding_id: string;
  checklist_item_id?: string;
  severity: Severity;
  title: string;
  description: string;
  confidence_score: number;
  source_doc_id?: string;
  source_page?: number;
  conflicting_doc_id?: string;
  conflicting_page?: number;
  evidence: string[];
  recommendation?: string;
}

export interface ReconciliationSnapshot {
  currency: string;
  contract_total: number | null;
  invoice_total: number | null;
  bank_paid_total: number | null;
  variance_vs_contract: number | null;
  notes?: string | null;
  /** True when extracted bank debits don't reconcile with the statement's stated Total Debits. */
  bank_extraction_incomplete?: boolean;
  bank_debits_verified?: number | null;
  bank_debits_stated?: number | null;
}

export interface EntityConflictRow {
  entity_label: string;
  doc_a_id: string;
  doc_a_value: string;
  doc_b_id: string;
  doc_b_value: string;
  severity: Severity;
  /** Human-readable reason the two values were compared, e.g. "contract total vs invoice total". */
  conflict_type: string;
  anchor_hint?: string | null;
}

export interface AuditReport {
  audit_id: string;
  title: string;
  language: "arabic" | "english";
  executive_summary: string;
  documents_reviewed: string[];
  findings: Finding[];
  recommendations: string[];
  overall_risk: OverallRisk;
  generated_at: string;
  reconciliation_snapshot?: ReconciliationSnapshot | null;
  entity_conflicts?: EntityConflictRow[];
}

// ─── Finding Triage ──────────────────────────────────────────────────────────

export type TriageStatus = "accepted" | "dismissed" | "false_positive";

export interface TriageRecord {
  status: TriageStatus;
  note?: string | null;
  signature?: string | null;
  updated_at: string;
}

export type TriageMap = Record<string, TriageRecord>;

// ─── Conversational Q&A ──────────────────────────────────────────────────────

export interface ChatSource {
  doc_id: string;
  doc_name: string;
  page: number;
  text: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
}

// ─── Reasoning Trace ─────────────────────────────────────────────────────────

export type AgentName = "extraction" | "planner" | "cross_checker" | "report_writer";
export type StepType = "thought" | "tool_call" | "tool_result" | "finding" | "summary";

export interface ReasoningStep {
  step_id: string;
  agent: AgentName;
  step_type: StepType;
  content: string;
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_output?: string;
  timestamp: string;
}

// ─── SSE Event Types ─────────────────────────────────────────────────────────

export type SSEEventType =
  | "connected"
  | "reasoning_step"
  | "finding"
  | "report_ready"
  | "complete"
  | "error";

export interface SSEEvent {
  type: SSEEventType;
  audit_id?: string;
  step?: ReasoningStep;
  overall_risk?: OverallRisk;
  finding_count?: number;
  document_count?: number;
  message?: string;
}

// ─── Upload ───────────────────────────────────────────────────────────────────

export interface UploadFileError {
  filename: string;
  error: string;
}

export interface UploadResponse {
  audit_id: string;
  documents: DocumentMeta[];
  message: string;
  failed: UploadFileError[];
}

// ─── Audit Archive ────────────────────────────────────────────────────────────

export interface AuditSummary {
  audit_id: string;
  status: AuditStatus;
  document_count: number;
  filenames: string[];
  created_at: string;
  completed_at?: string | null;
  error?: string | null;
  overall_risk?: OverallRisk | null;
  critical_count: number;
  warning_count: number;
}

// ─── Hook Return Types ────────────────────────────────────────────────────────

export interface SessionDocumentRef {
  doc_id: string;
  filename: string;
  doc_type: string;
}

export interface AuditStreamState {
  steps: ReasoningStep[];
  findings: Finding[];
  report: AuditReport | null;
  sessionDocuments: SessionDocumentRef[];
  isRunning: boolean;
  isComplete: boolean;
  error: string | null;
  agentStatuses: Record<AgentName, "pending" | "active" | "done">;
}
