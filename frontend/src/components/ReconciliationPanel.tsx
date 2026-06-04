"use client";

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import type { AuditReport, EntityConflictRow } from "@/types";

function formatMoney(n: number | null | undefined, currency: string): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 })} ${currency}`;
}

type SortKey = "entity" | "severity" | "doc_a" | "doc_b";

function shortDocLabel(id: string, docFilenameById: Record<string, string>): string {
  return docFilenameById[id] || `${id.slice(0, 8)}…`;
}

const CONFLICT_BADGE: Record<string, string> = {
  critical: "bg-red-50 text-red-700 border-red-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  ok: "bg-emerald-50 text-emerald-700 border-emerald-200",
};

export function ReconciliationPanel({
  report,
  isComplete,
  isRunning,
  docFilenameById = {},
}: {
  report: AuditReport | null;
  isComplete: boolean;
  isRunning: boolean;
  docFilenameById?: Record<string, string>;
}) {
  const snap = report?.reconciliation_snapshot;
  const rows = report?.entity_conflicts ?? [];

  const [sortKey, setSortKey] = useState<SortKey>("severity");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const sortedRows = useMemo(() => {
    const copy = [...rows];
    const dir = sortDir === "asc" ? 1 : -1;
    copy.sort((a, b) => {
      let cmp = 0;
      if (sortKey === "severity") {
        const rank = (s: string) => (s === "critical" ? 2 : s === "warning" ? 1 : 0);
        cmp = rank(a.severity) - rank(b.severity);
      } else if (sortKey === "entity") {
        cmp = a.entity_label.localeCompare(b.entity_label);
      } else if (sortKey === "doc_a") {
        cmp = a.doc_a_value.localeCompare(b.doc_a_value);
      } else {
        cmp = a.doc_b_value.localeCompare(b.doc_b_value);
      }
      return cmp * dir;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const variance = snap?.variance_vs_contract;
  const varianceIsFlag = variance !== null && variance !== undefined && !Number.isNaN(variance) && Math.abs(variance) > 1;

  if (!isComplete && isRunning && !report) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center text-sm text-muted-foreground bg-white border border-outline-variant rounded-lg">
        <span className="material-symbols-outlined text-[40px] text-muted-foreground/40 mb-3">balance</span>
        Reconciliation data will be available when the audit completes.
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl">

      {/* Metrics grid */}
      {snap ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Contract Total", value: formatMoney(snap.contract_total, snap.currency), icon: "receipt_long", variant: "default" },
            { label: "Invoice Total", value: formatMoney(snap.invoice_total, snap.currency), icon: "description", variant: "default" },
            { label: "Bank Paid Total", value: formatMoney(snap.bank_paid_total, snap.currency), icon: "account_balance", variant: "default" },
            {
              label: "Variance Flagged",
              value: variance === null || variance === undefined
                ? "—"
                : `${variance >= 0 ? "+" : ""}${variance.toLocaleString()} ${snap.currency}`,
              icon: "warning",
              variant: varianceIsFlag ? "error" : "ok",
            },
          ].map(({ label, value, icon, variant }) => (
            <div
              key={label}
              className={cn(
                "rounded-xl border p-4",
                variant === "error" && "bg-red-50 border-red-200",
                variant === "ok" && "bg-emerald-50 border-emerald-200",
                variant === "default" && "bg-white border-outline-variant"
              )}
            >
              <div className="flex items-center gap-2 mb-2">
                <span className={cn(
                  "material-symbols-outlined text-[18px]",
                  variant === "error" && "text-red-600",
                  variant === "ok" && "text-emerald-600",
                  variant === "default" && "text-muted-foreground"
                )}>{icon}</span>
                <p className={cn(
                  "text-[11px] font-semibold uppercase tracking-wide",
                  variant === "error" && "text-red-700",
                  variant === "ok" && "text-emerald-700",
                  variant === "default" && "text-muted-foreground"
                )}>{label}</p>
              </div>
              <p className={cn(
                "text-xl font-bold tabular-nums",
                variant === "error" && "text-red-700",
                variant === "ok" && "text-emerald-700",
                variant === "default" && "text-foreground"
              )}>{value}</p>
            </div>
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-outline-variant bg-secondary px-4 py-8 text-center text-sm text-muted-foreground">
          {isComplete
            ? "Insufficient structured data to build a reconciliation snapshot for this audit."
            : "No snapshot yet."}
        </div>
      )}

      {snap?.notes && (
        <p className="text-xs text-muted-foreground px-1">{snap.notes}</p>
      )}

      {/* Entity conflicts table */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-bold text-foreground flex items-center gap-2">
              <span className="material-symbols-outlined text-[18px] text-primary">compare_arrows</span>
              Entity Conflicts
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              Cross-document conflicts detected in the knowledge graph.
            </p>
          </div>
          <div className="flex gap-2">
            <button className="flex items-center gap-1.5 px-3 py-1.5 border border-outline-variant rounded text-xs font-semibold text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors">
              <span className="material-symbols-outlined text-[14px]">upload</span>
              Export
            </button>
            <button className="flex items-center gap-1.5 px-3 py-1.5 border border-outline-variant rounded text-xs font-semibold text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors">
              <span className="material-symbols-outlined text-[14px]">picture_as_pdf</span>
              Download PDF
            </button>
          </div>
        </div>

        {sortedRows.length === 0 ? (
          <div className="rounded-lg border border-outline-variant bg-white px-4 py-10 text-center text-sm text-muted-foreground">
            No graph-linked amount conflicts for this session.
          </div>
        ) : (
          <div className="bg-white rounded-lg border border-outline-variant overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-secondary border-b border-outline-variant">
                  {[
                    { key: "entity" as SortKey, label: "Entity ID" },
                    { key: "doc_a" as SortKey, label: "Doc A (Contract)" },
                    { key: "doc_b" as SortKey, label: "Doc B (Invoice)" },
                    { key: "severity" as SortKey, label: "Conflict Type" },
                  ].map(({ key, label }) => (
                    <th key={key} className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
                        onClick={() => toggleSort(key)}
                      >
                        {label}
                        {sortKey === key && (
                          <span className="material-symbols-outlined text-[12px]">
                            {sortDir === "asc" ? "arrow_upward" : "arrow_downward"}
                          </span>
                        )}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((r: EntityConflictRow, i: number) => {
                  const docAName = shortDocLabel(r.doc_a_id, docFilenameById);
                  const docBName = shortDocLabel(r.doc_b_id, docFilenameById);
                  return (
                    <tr key={i} className="border-b border-outline-variant last:border-0 hover:bg-secondary transition-colors">
                      <td className="py-3 px-4 align-top">
                        <p className="text-[13px] font-medium text-foreground max-w-[180px] truncate">{r.entity_label}</p>
                        {r.anchor_hint && (
                          <p className="text-[11px] text-muted-foreground mt-0.5 truncate">{r.anchor_hint}</p>
                        )}
                      </td>
                      <td className="py-3 px-4 align-top">
                        <p className="text-[11px] text-muted-foreground mb-0.5 truncate max-w-[150px]">{docAName}</p>
                        <p className="text-[13px] font-mono text-foreground">{r.doc_a_value}</p>
                      </td>
                      <td className="py-3 px-4 align-top">
                        <p className="text-[11px] text-muted-foreground mb-0.5 truncate max-w-[150px]">{docBName}</p>
                        <p className="text-[13px] font-mono text-foreground">{r.doc_b_value}</p>
                      </td>
                      <td className="py-3 px-4 align-top">
                        <span className={cn(
                          "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[11px] font-semibold",
                          CONFLICT_BADGE[r.severity] ?? "bg-secondary text-foreground border-outline-variant"
                        )}>
                          <span className="material-symbols-outlined text-[11px]">
                            {r.severity === "critical" ? "error" : r.severity === "warning" ? "warning" : "check_circle"}
                          </span>
                          {r.severity.charAt(0).toUpperCase() + r.severity.slice(1)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

