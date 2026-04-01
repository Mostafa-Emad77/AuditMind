import type { AuditReport, UploadResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function uploadDocuments(
  files: File[],
  reportLanguage: "arabic" | "english" = "english"
): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const res = await fetch(
    `${API_BASE}/api/upload?report_language=${reportLanguage}`,
    { method: "POST", body: formData }
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

export async function getAuditReport(auditId: string): Promise<AuditReport> {
  const res = await fetch(`${API_BASE}/api/audit/${auditId}/report`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Report not found" }));
    throw new Error(err.detail || "Failed to fetch report");
  }
  return res.json();
}

export async function getAuditStatus(auditId: string) {
  const res = await fetch(`${API_BASE}/api/audit/${auditId}/status`);
  if (!res.ok) throw new Error("Failed to fetch audit status");
  return res.json();
}

export function createAuditEventSource(auditId: string): EventSource {
  return new EventSource(`${API_BASE}/api/audit/${auditId}/stream`);
}
