import type {
  AuditReport,
  AuditSummary,
  ChatMessage,
  ChatSource,
  TriageMap,
  TriageRecord,
  TriageStatus,
  UploadResponse,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

/** Merge the API key header into a fetch init, if one is configured. */
function withAuth(init: RequestInit = {}): RequestInit {
  if (!API_KEY) return init;
  return { ...init, headers: { ...init.headers, "X-API-Key": API_KEY } };
}

/** EventSource can't set headers, so the key travels as a query param there. */
function withAuthQuery(url: string): string {
  if (!API_KEY) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}api_key=${encodeURIComponent(API_KEY)}`;
}

export async function uploadDocuments(
  files: File[],
  reportLanguage: "arabic" | "english" = "english"
): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const res = await fetch(
    `${API_BASE}/api/upload?report_language=${reportLanguage}`,
    withAuth({ method: "POST", body: formData })
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

export async function getAuditReport(auditId: string): Promise<AuditReport> {
  const res = await fetch(`${API_BASE}/api/audit/${auditId}/report`, withAuth());
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Report not found" }));
    throw new Error(err.detail || "Failed to fetch report");
  }
  return res.json();
}

export async function listAudits(): Promise<AuditSummary[]> {
  const res = await fetch(`${API_BASE}/api/audits`, withAuth());
  if (!res.ok) throw new Error("Failed to fetch audit history");
  const data = await res.json().catch(() => ({ audits: [] }));
  return (data.audits ?? []) as AuditSummary[];
}

export async function getAuditStatus(auditId: string) {
  const res = await fetch(`${API_BASE}/api/audit/${auditId}/status`, withAuth());
  if (!res.ok) throw new Error("Failed to fetch audit status");
  return res.json();
}

export function createAuditEventSource(auditId: string): EventSource {
  return new EventSource(withAuthQuery(`${API_BASE}/api/audit/${auditId}/stream`));
}

// ─── Conversational Q&A ──────────────────────────────────────────────────────

export interface ChatStreamHandlers {
  onSources?: (sources: ChatSource[]) => void;
  onDelta?: (text: string) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
}

/** POST a question and stream the grounded answer (SSE over fetch). */
export async function streamChat(
  auditId: string,
  message: string,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/audit/${auditId}/chat`,
    withAuth({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal,
    })
  );

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: "Chat request failed" }));
    throw new Error(err.detail || "Chat request failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      const payload = dataLine.slice(5).trim();
      if (!payload) continue;
      try {
        const evt = JSON.parse(payload);
        switch (evt.type) {
          case "sources":
            handlers.onSources?.(evt.sources ?? []);
            break;
          case "delta":
            handlers.onDelta?.(evt.text ?? "");
            break;
          case "done":
            handlers.onDone?.();
            break;
          case "error":
            handlers.onError?.(evt.message ?? "An error occurred");
            break;
        }
      } catch {
        // ignore malformed frames
      }
    }
  }
}

export async function getChatHistory(auditId: string): Promise<ChatMessage[]> {
  const res = await fetch(`${API_BASE}/api/audit/${auditId}/chat`, withAuth());
  if (!res.ok) return [];
  const data = await res.json().catch(() => ({ messages: [] }));
  return (data.messages ?? []) as ChatMessage[];
}

// ─── Finding triage ──────────────────────────────────────────────────────────

export async function getTriage(auditId: string): Promise<TriageMap> {
  const res = await fetch(`${API_BASE}/api/audit/${auditId}/triage`, withAuth());
  if (!res.ok) return {};
  const data = await res.json().catch(() => ({ triage: {} }));
  return (data.triage ?? {}) as TriageMap;
}

export async function triageFinding(
  auditId: string,
  findingId: string,
  status: TriageStatus,
  note?: string
): Promise<TriageRecord> {
  const res = await fetch(
    `${API_BASE}/api/audit/${auditId}/findings/${findingId}/triage`,
    withAuth({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, note }),
    })
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Triage failed" }));
    throw new Error(err.detail || "Failed to record triage");
  }
  const data = await res.json();
  return data.triage as TriageRecord;
}
