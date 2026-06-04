"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Finding, Severity } from "@/types";

/* ── severity helpers ── */
const SEVERITY_BADGE: Record<Severity, string> = {
  critical: "bg-red-50 text-red-700 border-red-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  ok: "bg-emerald-50 text-emerald-700 border-emerald-200",
};

const SEVERITY_ROW_SELECTED: Record<Severity, string> = {
  critical: "bg-red-50/60",
  warning: "bg-amber-50/60",
  ok: "bg-emerald-50/60",
};

const SEVERITY_ICON: Record<Severity, string> = {
  critical: "error",
  warning: "warning",
  ok: "check_circle",
};

/* ── individual row ── */
interface FindingRowProps {
  finding: Finding;
  index: number;
  onOpenDetail?: (f: Finding) => void;
}

function FindingRow({ finding, index, onOpenDetail }: FindingRowProps) {
  const [isExpanded, setIsExpanded] = useState(finding.severity === "critical");

  return (
    <motion.tr
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: index * 0.04 }}
      className={cn(
        "border-b border-outline-variant hover:bg-secondary transition-colors cursor-pointer",
        isExpanded && SEVERITY_ROW_SELECTED[finding.severity]
      )}
    >
      {/* Expand / detail in a nested structure */}
      <td colSpan={5} className="p-0">
        {/* Header row */}
        <div
          className="flex items-center gap-0 cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {/* Priority badge */}
          <div className="py-2 px-4 w-[130px] shrink-0">
            <span className={cn(
              "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[11px] font-bold",
              SEVERITY_BADGE[finding.severity]
            )}>
              <span className="material-symbols-outlined text-[12px]">{SEVERITY_ICON[finding.severity]}</span>
              {finding.severity.charAt(0).toUpperCase() + finding.severity.slice(1)}
            </span>
          </div>
          {/* Description */}
          <div className="flex-1 py-2 px-4 text-[13px] text-foreground font-medium min-w-0">
            {finding.title}
          </div>
          {/* Confidence */}
          <div className="py-2 px-4 w-[100px] text-right font-mono text-[13px] font-medium text-primary tabular-nums shrink-0">
            {Math.round(finding.confidence_score * 100)}%
          </div>
          {/* Source */}
          <div className="py-2 px-4 w-[160px] shrink-0">
            {finding.source_doc_id && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-muted text-foreground text-[11px] rounded border border-outline-variant">
                <span className="material-symbols-outlined text-[12px]">description</span>
                {finding.source_doc_id.slice(0, 8)}…
              </span>
            )}
          </div>
          {/* Detail button */}
          <div className="py-2 px-4 w-10 text-right shrink-0">
            {onOpenDetail ? (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onOpenDetail(finding); }}
                className="text-primary hover:text-[#003ea8] transition-colors"
              >
                <span className="material-symbols-outlined text-[20px]">chevron_right</span>
              </button>
            ) : (
              <span className="material-symbols-outlined text-[20px] text-outline-variant">
                {isExpanded ? "expand_less" : "expand_more"}
              </span>
            )}
          </div>
        </div>

        {/* Expanded details */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden border-t border-outline-variant"
            >
              <div className="px-4 pb-4 pt-3 ml-[130px] space-y-3">
                <p
                  className="text-[13px] text-on-surface-variant leading-relaxed"
                  dir={finding.description.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                >
                  {finding.description}
                </p>
                {finding.evidence.length > 0 && (
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mb-2">Evidence</p>
                    <div className="space-y-1.5">
                      {finding.evidence.map((e, i) => (
                        <div key={i} className="flex items-start gap-2 text-[12px] bg-muted/50 border border-outline-variant px-3 py-2 rounded">
                          <span className="material-symbols-outlined text-[13px] text-muted-foreground mt-0.5">description</span>
                          <span className="text-muted-foreground" dir={e.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}>{e}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {finding.recommendation && (
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-50 border border-blue-100">
                    <span className="material-symbols-outlined text-[16px] text-primary shrink-0 mt-0.5">smart_toy</span>
                    <p
                      className="text-[12px] text-foreground"
                      dir={finding.recommendation.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                    >
                      {finding.recommendation}
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </td>
    </motion.tr>
  );
}

/* ── main table ── */
interface FindingsTableProps {
  findings: Finding[];
  isLoading?: boolean;
  onOpenDetail?: (finding: Finding) => void;
}

export function FindingsTable({ findings, isLoading, onOpenDetail }: FindingsTableProps) {
  const [filter, setFilter] = useState<Severity | "all">("all");

  const critical = findings.filter((f) => f.severity === "critical");
  const warnings = findings.filter((f) => f.severity === "warning");
  const ok = findings.filter((f) => f.severity === "ok");
  const filtered = filter === "all" ? findings : findings.filter((f) => f.severity === filter);

  if (isLoading) {
    return (
      <div className="bg-white border border-outline-variant rounded-lg overflow-hidden">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-12 border-b border-outline-variant shimmer last:border-0" />
        ))}
      </div>
    );
  }

  if (findings.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center bg-white border border-outline-variant rounded-lg">
        <CheckCircle2 className="h-8 w-8 text-muted-foreground/40 mb-3" />
        <p className="text-sm text-muted-foreground">No findings yet</p>
        <p className="text-xs text-muted-foreground/60 mt-1">
          Findings will appear here as the audit progresses
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter tabs */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {[
          { label: "All", value: "all", count: findings.length },
          { label: "Critical", value: "critical", count: critical.length },
          { label: "Warning", value: "warning", count: warnings.length },
          { label: "Info", value: "ok", count: ok.length },
        ].map(({ label, value, count }) => (
          <button
            key={value}
            onClick={() => setFilter(value as typeof filter)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1 rounded text-xs font-semibold uppercase tracking-wide border transition-all",
              filter === value
                ? "bg-primary text-white border-primary"
                : "bg-white text-muted-foreground border-outline-variant hover:bg-secondary"
            )}
          >
            {count} {label}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="bg-white border border-outline-variant rounded-lg overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-secondary border-b border-outline-variant">
              <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground w-[130px]">Priority</th>
              <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Description</th>
              <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground text-right w-[100px]">Confidence</th>
              <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground w-[160px]">Source</th>
              <th className="py-3 px-4 w-10" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((finding, i) => (
              <FindingRow
                key={finding.finding_id}
                finding={finding}
                index={i}
                onOpenDetail={onOpenDetail}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

