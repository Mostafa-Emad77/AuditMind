"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { cn } from "@/lib/utils";
import type { Finding, TriageStatus } from "@/types";

/** Parse "filename, page N: snippet" style evidence lines. */
export function parseEvidenceLine(line: string): {
  filename: string;
  page?: number;
  quote: string;
} {
  const m = line.match(/^(.+?),\s*page\s+(\d+)\s*:\s*(.*)$/i);
  if (m) {
    return { filename: m[1].trim(), page: parseInt(m[2], 10), quote: m[3].trim() };
  }
  return { filename: "", quote: line.trim() };
}

function resolveFilename(raw: string, docFilenameById: Record<string, string>): string {
  const s = raw.trim();
  for (const [id, name] of Object.entries(docFilenameById)) {
    if (s.includes(id.slice(0, 8))) return name;
  }
  return raw;
}

const SEVERITY_HEADER: Record<string, string> = {
  critical: "bg-red-50 border-red-200",
  warning: "bg-amber-50 border-amber-200",
  ok: "bg-emerald-50 border-emerald-200",
};

const SEVERITY_BADGE: Record<string, string> = {
  critical: "bg-red-50 text-red-700 border-red-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  ok: "bg-emerald-50 text-emerald-700 border-emerald-200",
};

const TRIAGE_OPTIONS = [
  {
    status: "accepted" as const,
    label: "Accept",
    icon: "check_circle",
    active: "bg-emerald-600 text-white border-emerald-600",
    idle: "text-emerald-700 border-emerald-200 hover:bg-emerald-50",
  },
  {
    status: "dismissed" as const,
    label: "Dismiss",
    icon: "do_not_disturb_on",
    active: "bg-slate-600 text-white border-slate-600",
    idle: "text-slate-700 border-outline-variant hover:bg-secondary",
  },
  {
    status: "false_positive" as const,
    label: "False positive",
    icon: "flag",
    active: "bg-red-600 text-white border-red-600",
    idle: "text-red-700 border-red-200 hover:bg-red-50",
  },
];

export function FindingDetailSheet({
  finding,
  open,
  onOpenChange,
  docFilenameById,
  triageStatus,
  onTriage,
}: {
  finding: Finding | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  docFilenameById: Record<string, string>;
  triageStatus?: TriageStatus | null;
  onTriage?: (status: TriageStatus) => void;
}) {
  if (!finding) return null;

  const sourceDocName = finding.source_doc_id
    ? docFilenameById[finding.source_doc_id] || `${finding.source_doc_id.slice(0, 8)}…`
    : null;
  const conflictDocName = finding.conflicting_doc_id
    ? docFilenameById[finding.conflicting_doc_id] || `${finding.conflicting_doc_id.slice(0, 8)}…`
    : null;

  const hasConflict = finding.severity === "critical" && (finding.evidence.length >= 2 || conflictDocName);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/30 z-50" />
        <Dialog.Content className="fixed right-0 top-0 z-50 h-full w-full max-w-[460px] border-l border-outline-variant bg-white shadow-2xl flex flex-col outline-none">

          {/* Header */}
          <div className="flex items-center justify-between gap-3 px-5 py-4 border-b border-outline-variant">
            <div className="flex items-center gap-2.5 min-w-0">
              <span className="material-symbols-outlined text-[22px] text-primary shrink-0">hub</span>
              <div className="min-w-0">
                <Dialog.Title className="text-sm font-bold text-foreground leading-tight truncate">
                  Evidence Traceability
                </Dialog.Title>
                <p className="text-[11px] text-muted-foreground mt-0.5 truncate">{finding.title}</p>
              </div>
            </div>
            <Dialog.Close className="rounded p-1.5 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors shrink-0">
              <span className="material-symbols-outlined text-[20px]">close</span>
            </Dialog.Close>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-5">

            {/* Severity + conflict alert */}
            <div className={cn(
              "flex items-start gap-3 p-3 rounded-lg border",
              SEVERITY_HEADER[finding.severity] ?? "bg-secondary border-outline-variant"
            )}>
              <span className={cn(
                "material-symbols-outlined text-[20px] shrink-0 mt-0.5",
                finding.severity === "critical" && "text-red-600",
                finding.severity === "warning" && "text-amber-600",
                finding.severity === "ok" && "text-emerald-600"
              )}>
                {finding.severity === "critical" ? "error" : finding.severity === "warning" ? "warning" : "check_circle"}
              </span>
              <div>
                <p className="text-[12px] font-bold text-foreground">
                  {finding.severity === "critical" ? "Conflict Detected" :
                   finding.severity === "warning" ? "Discrepancy Found" :
                   "No Conflict Found"}
                </p>
                <p className="text-[11px] text-muted-foreground mt-0.5 leading-snug">
                  {finding.description.slice(0, 140)}{finding.description.length > 140 ? "…" : ""}
                </p>
              </div>
            </div>

            {/* Source comparison */}
            {finding.evidence.length >= 2 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-3">Source Comparison</p>
                <div className="relative flex flex-col gap-0">
                  {/* Vertical connector */}
                  <div className="absolute left-[18px] top-12 bottom-12 w-0.5 bg-outline-variant" />

                  {[0, 1].map((idx) => {
                    const line = finding.evidence[idx];
                    const parsed = parseEvidenceLine(line);
                    const label = `Source ${String.fromCharCode(65 + idx)}`;
                    const fname = parsed.filename
                      ? resolveFilename(parsed.filename, docFilenameById)
                      : idx === 0
                      ? sourceDocName ?? "Document A"
                      : conflictDocName ?? "Document B";

                    return (
                      <div key={idx} className="relative flex gap-3 p-3 bg-secondary rounded-lg border border-outline-variant mb-2 last:mb-0 z-10">
                        <div className="w-8 h-8 rounded bg-white border border-outline-variant flex items-center justify-center shrink-0">
                          <span className="material-symbols-outlined text-[14px] text-primary">description</span>
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[10px] font-bold text-muted-foreground uppercase">{label}</span>
                            {fname && (
                              <span className="text-[10px] text-muted-foreground truncate max-w-[180px]">{fname}</span>
                            )}
                            {parsed.page != null && (
                              <span className="text-[10px] bg-muted border border-outline-variant px-1.5 py-0.5 rounded text-muted-foreground">p.{parsed.page}</span>
                            )}
                          </div>
                          <p
                            className="text-[12px] text-foreground leading-relaxed"
                            dir={parsed.quote.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                          >
                            {parsed.quote || line}
                          </p>
                        </div>
                      </div>
                    );
                  })}

                  {finding.evidence.length > 2 && (
                    <p className="text-[11px] text-muted-foreground mt-1">+ {finding.evidence.length - 2} more source{finding.evidence.length - 2 > 1 ? "s" : ""}</p>
                  )}
                </div>
              </div>
            )}

            {/* Single evidence */}
            {finding.evidence.length === 1 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">Source Passage</p>
                {(() => {
                  const parsed = parseEvidenceLine(finding.evidence[0]);
                  const fname = parsed.filename
                    ? resolveFilename(parsed.filename, docFilenameById)
                    : sourceDocName ?? "Document";
                  return (
                    <div className="flex gap-3 p-3 bg-secondary rounded-lg border border-outline-variant">
                      <span className="material-symbols-outlined text-[16px] text-primary shrink-0 mt-0.5">description</span>
                      <div>
                        <p className="text-[11px] font-semibold text-muted-foreground mb-1">
                          {fname}{parsed.page != null ? ` · p.${parsed.page}` : ""}
                        </p>
                        <p
                          className="text-[12px] text-foreground"
                          dir={parsed.quote.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                        >
                          {parsed.quote || finding.evidence[0]}
                        </p>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* Recommendation / Suggested Action */}
            {finding.recommendation && (
              <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="material-symbols-outlined text-[16px] text-primary">smart_toy</span>
                  <p className="text-[11px] font-bold uppercase tracking-widest text-primary">Suggested Auditor Action</p>
                </div>
                <p
                  className="text-[12px] text-foreground leading-relaxed"
                  dir={finding.recommendation.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                >
                  {finding.recommendation}
                </p>
                {finding.severity === "critical" && (
                  <div className="mt-3 flex gap-2">
                    <button className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 text-[11px] font-semibold text-primary border border-primary rounded hover:bg-primary hover:text-white transition-colors">
                      <span className="material-symbols-outlined text-[13px]">mail</span>
                      Generate RFI Email
                    </button>
                    <button className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 text-[11px] font-semibold text-red-600 border border-red-200 bg-red-50 rounded hover:bg-red-100 transition-colors">
                      <span className="material-symbols-outlined text-[13px]">flag</span>
                      Flag as Material
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Reviewer verdict (triage) */}
            {onTriage && (
              <div className="border-t border-outline-variant pt-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">
                  Reviewer Verdict
                </p>
                <div className="flex gap-2">
                  {TRIAGE_OPTIONS.map(({ status, label, icon, active, idle }) => (
                    <button
                      key={status}
                      type="button"
                      onClick={() => onTriage(status)}
                      className={cn(
                        "flex-1 flex items-center justify-center gap-1.5 py-2 px-2 text-[11px] font-semibold border rounded transition-colors",
                        triageStatus === status ? active : idle
                      )}
                    >
                      <span className="material-symbols-outlined text-[14px]">{icon}</span>
                      {label}
                    </button>
                  ))}
                </div>
                {triageStatus === "false_positive" && (
                  <p className="text-[10px] text-muted-foreground mt-2">
                    This pattern will be suppressed in future audit runs.
                  </p>
                )}
              </div>
            )}

            {/* Confidence */}
            <div className="flex items-center justify-between py-3 border-t border-outline-variant">
              <span className="text-[11px] text-muted-foreground font-semibold uppercase tracking-wide">AI Confidence</span>
              <div className="flex items-center gap-2">
                <div className="w-24 h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full"
                    style={{ width: `${Math.round(finding.confidence_score * 100)}%` }}
                  />
                </div>
                <span className="font-mono text-[13px] font-bold text-primary tabular-nums">
                  {Math.round(finding.confidence_score * 100)}%
                </span>
              </div>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

