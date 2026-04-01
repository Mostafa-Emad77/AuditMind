"use client";

import { useMemo, useState } from "react";
import { ArrowDownUp, Scale } from "lucide-react";
import type { AuditReport, EntityConflictRow } from "@/types";
import { cn, severityBg } from "@/lib/utils";

function formatMoney(n: number | null | undefined, currency: string): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 })} ${currency}`;
}

type SortKey = "entity" | "severity" | "doc_a" | "doc_b";

function shortDocLabel(id: string, docFilenameById: Record<string, string>): string {
  return docFilenameById[id] || `${id.slice(0, 8)}…`;
}

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
  const varianceTone =
    variance === null || variance === undefined
      ? "text-muted-foreground"
      : Math.abs(variance) < 1
        ? "text-emerald-400"
        : variance > 0
          ? "text-red-400"
          : "text-amber-400";

  if (!isComplete && isRunning && !report) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center text-sm text-muted-foreground">
        <Scale className="h-10 w-10 mb-3 opacity-30" />
        Reconciliation data will be available when the audit completes.
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-4xl">
      <div>
        <h2 className="text-sm font-semibold mb-1 flex items-center gap-2">
          <Scale className="h-4 w-4 text-primary" />
          Number reconciliation
        </h2>
        <p className="text-xs text-muted-foreground mb-4">
          Cross-document totals derived from extracted entities and graph-linked amounts.
        </p>

        {!snap ? (
          <div className="rounded-xl border border-dashed border-border bg-secondary/20 px-4 py-8 text-center text-sm text-muted-foreground">
            {isComplete
              ? "Insufficient structured data to build a reconciliation snapshot for this audit."
              : "No snapshot yet."}
          </div>
        ) : (
          <div className="rounded-xl border border-border bg-card/40 overflow-hidden">
            <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-border/60 border-b border-border/60">
              <div className="p-4">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">
                  Contract total
                </p>
                <p className="text-lg font-semibold mt-1 tabular-nums">
                  {formatMoney(snap.contract_total, snap.currency)}
                </p>
              </div>
              <div className="p-4">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">
                  Invoice total
                </p>
                <p className="text-lg font-semibold mt-1 tabular-nums">
                  {formatMoney(snap.invoice_total, snap.currency)}
                </p>
              </div>
              <div className="p-4">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">
                  Bank paid (sum)
                </p>
                <p className="text-lg font-semibold mt-1 tabular-nums">
                  {formatMoney(snap.bank_paid_total, snap.currency)}
                </p>
              </div>
              <div className="p-4 bg-primary/5">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">
                  vs contract (gap)
                </p>
                <p className={cn("text-lg font-semibold mt-1 tabular-nums", varianceTone)}>
                  {variance === null || variance === undefined ? "—" : `${variance >= 0 ? "+" : ""}${variance.toLocaleString()} ${snap.currency}`}
                </p>
              </div>
            </div>
            {snap.notes && (
              <p className="text-xs text-muted-foreground px-4 py-2 border-t border-border/40 bg-secondary/20">
                {snap.notes}
              </p>
            )}
          </div>
        )}
      </div>

      <div>
        <h2 className="text-sm font-semibold mb-1 flex items-center gap-2">
          <ArrowDownUp className="h-4 w-4 text-primary" />
          Entity conflicts
        </h2>
        <p className="text-xs text-muted-foreground mb-3">
          Only cross-document conflicts detected in the knowledge graph (not all extracted entities).
        </p>

        {sortedRows.length === 0 ? (
          <div className="rounded-lg border border-border bg-secondary/10 px-4 py-6 text-center text-sm text-muted-foreground">
            No graph-linked amount conflicts for this session.
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-secondary/30 text-left text-xs text-muted-foreground">
                  <th className="p-2 font-medium">
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      onClick={() => toggleSort("entity")}
                    >
                      Entity / anchor
                    </button>
                  </th>
                  <th className="p-2 font-medium">
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      onClick={() => toggleSort("doc_a")}
                    >
                      Document A
                    </button>
                  </th>
                  <th className="p-2 font-medium">
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      onClick={() => toggleSort("doc_b")}
                    >
                      Document B
                    </button>
                  </th>
                  <th className="p-2 font-medium w-28">
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      onClick={() => toggleSort("severity")}
                    >
                      Severity
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((r: EntityConflictRow, i: number) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-accent/20">
                    <td className="p-2 align-top max-w-[200px]">
                      <span className="font-medium text-foreground/90 line-clamp-2">{r.entity_label}</span>
                      {r.anchor_hint && (
                        <span className="block text-[10px] text-muted-foreground mt-0.5 truncate">
                          {r.anchor_hint}
                        </span>
                      )}
                    </td>
                    <td className="p-2 align-top text-xs max-w-[200px]">
                      <span className="text-[10px] text-muted-foreground block mb-0.5 truncate">
                        {shortDocLabel(r.doc_a_id, docFilenameById)}
                      </span>
                      <span className="font-mono break-all">{r.doc_a_value}</span>
                    </td>
                    <td className="p-2 align-top text-xs max-w-[200px]">
                      <span className="text-[10px] text-muted-foreground block mb-0.5 truncate">
                        {shortDocLabel(r.doc_b_id, docFilenameById)}
                      </span>
                      <span className="font-mono break-all">{r.doc_b_value}</span>
                    </td>
                    <td className="p-2 align-top">
                      <span
                        className={cn(
                          "text-[10px] font-semibold px-2 py-0.5 rounded-full border",
                          severityBg(r.severity)
                        )}
                      >
                        {r.severity}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
