"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  FileText,
  PanelRight,
} from "lucide-react";
import { cn, severityBg } from "@/lib/utils";
import type { Finding, Severity } from "@/types";

const SEVERITY_ICONS: Record<Severity, React.ReactNode> = {
  critical: <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />,
  warning: <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />,
  ok: <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />,
};

interface FindingRowProps {
  finding: Finding;
  index: number;
  onOpenDetail?: (f: Finding) => void;
}

function FindingRow({ finding, index, onOpenDetail }: FindingRowProps) {
  const [isExpanded, setIsExpanded] = useState(finding.severity === "critical");

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: index * 0.05 }}
      className={cn(
        "rounded-lg border overflow-hidden transition-all",
        finding.severity === "critical" && "border-red-400/30",
        finding.severity === "warning" && "border-amber-400/30",
        finding.severity === "ok" && "border-emerald-400/30"
      )}
    >
      {/* Header row */}
      <div className="flex items-stretch">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className={cn(
            "flex-1 flex items-start gap-3 p-3 text-left transition-colors hover:bg-accent/30",
            finding.severity === "critical" && "bg-red-400/5",
            finding.severity === "warning" && "bg-amber-400/5",
            finding.severity === "ok" && "bg-emerald-400/5"
          )}
        >
          {SEVERITY_ICONS[finding.severity]}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm font-medium leading-tight">{finding.title}</p>
              <div className="flex items-center gap-2 shrink-0">
                <span
                  className={cn(
                    "text-[10px] font-semibold px-2 py-0.5 rounded-full border",
                    severityBg(finding.severity)
                  )}
                >
                  {finding.severity.toUpperCase()}
                </span>
                <span className="text-[10px] text-muted-foreground">
                  {Math.round(finding.confidence_score * 100)}% confidence
                </span>
              </div>
            </div>
          </div>
          {isExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
          )}
        </button>
        {onOpenDetail && (
          <button
            type="button"
            title="Evidence and sources"
            className="px-3 border-l border-border/50 hover:bg-accent/50 text-muted-foreground hover:text-foreground shrink-0 flex items-center justify-center"
            onClick={() => onOpenDetail(finding)}
          >
            <PanelRight className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Expanded details */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-2 space-y-3 border-t border-border/50">
              {/* Description */}
              <p
                className="text-sm text-foreground/80 leading-relaxed"
                dir={finding.description.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
              >
                {finding.description}
              </p>

              {/* Evidence */}
              {finding.evidence.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1.5">Evidence:</p>
                  <div className="space-y-1">
                    {finding.evidence.map((e, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 text-xs bg-secondary/40 px-2.5 py-1.5 rounded"
                      >
                        <FileText className="h-3 w-3 text-muted-foreground mt-0.5 shrink-0" />
                        <span
                          className="text-foreground/70"
                          dir={e.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                        >
                          {e}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Source citations */}
              {(finding.source_doc_id || finding.conflicting_doc_id) && (
                <div className="flex flex-wrap gap-2 text-[10px]">
                  {finding.source_doc_id && (
                    <span className="px-2 py-1 bg-secondary rounded border border-border text-muted-foreground font-mono">
                      Doc: {finding.source_doc_id.slice(0, 8)}...{finding.source_page ? ` p.${finding.source_page}` : ""}
                    </span>
                  )}
                  {finding.conflicting_doc_id && (
                    <span className="px-2 py-1 bg-secondary rounded border border-border text-muted-foreground font-mono">
                      Conflicts with: {finding.conflicting_doc_id.slice(0, 8)}...{finding.conflicting_page ? ` p.${finding.conflicting_page}` : ""}
                    </span>
                  )}
                </div>
              )}

              {/* Recommendation */}
              {finding.recommendation && (
                <div className="flex items-start gap-2 p-2.5 rounded bg-primary/5 border border-primary/20">
                  <span className="text-primary text-xs shrink-0">💡</span>
                  <p
                    className="text-xs text-primary/80"
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
    </motion.div>
  );
}

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

  const filtered =
    filter === "all" ? findings : findings.filter((f) => f.severity === filter);

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-16 rounded-lg shimmer" />
        ))}
      </div>
    );
  }

  if (findings.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <CheckCircle2 className="h-8 w-8 text-muted-foreground/30 mb-3" />
        <p className="text-sm text-muted-foreground">No findings yet</p>
        <p className="text-xs text-muted-foreground/60 mt-1">
          Findings will appear here as the audit progresses
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Summary stats */}
      <div className="flex items-center gap-2 flex-wrap">
        {[
          { label: "All", value: "all", count: findings.length, color: "text-foreground" },
          { label: "Critical", value: "critical", count: critical.length, color: "text-red-400" },
          { label: "Warning", value: "warning", count: warnings.length, color: "text-amber-400" },
          { label: "OK", value: "ok", count: ok.length, color: "text-emerald-400" },
        ].map(({ label, value, count, color }) => (
          <button
            key={value}
            onClick={() => setFilter(value as typeof filter)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border transition-all",
              filter === value
                ? "bg-primary/10 border-primary/30 text-primary"
                : "bg-secondary border-border text-muted-foreground hover:border-primary/30"
            )}
          >
            <span className={color}>{count}</span>
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Finding rows */}
      <div className="space-y-2">
        {filtered.map((finding, i) => (
          <FindingRow
            key={finding.finding_id}
            finding={finding}
            index={i}
            onOpenDetail={onOpenDetail}
          />
        ))}
      </div>
    </div>
  );
}
