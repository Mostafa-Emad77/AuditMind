"use client";

import { motion } from "framer-motion";
import { cn, formatDate } from "@/lib/utils";
import type { AuditReport as AuditReportType } from "@/types";

const RISK_CONFIG: Record<string, { icon: string; badge: string; label: string }> = {
  critical: { icon: "error", badge: "bg-red-50 text-red-700 border-red-200", label: "CRITICAL RISK" },
  high: { icon: "warning", badge: "bg-orange-50 text-orange-700 border-orange-200", label: "HIGH RISK" },
  medium: { icon: "warning", badge: "bg-amber-50 text-amber-700 border-amber-200", label: "MEDIUM RISK" },
  low: { icon: "info", badge: "bg-blue-50 text-blue-700 border-blue-100", label: "LOW RISK" },
  clean: { icon: "check_circle", badge: "bg-emerald-50 text-emerald-700 border-emerald-200", label: "CLEAN" },
};

interface AuditReportProps {
  report: AuditReportType;
}

export function AuditReport({ report }: AuditReportProps) {
  const risk = RISK_CONFIG[report.overall_risk] ?? RISK_CONFIG.medium;
  const critical = report.findings.filter((f) => f.severity === "critical");
  const warnings = report.findings.filter((f) => f.severity === "warning");
  const ok = report.findings.filter((f) => f.severity === "ok");
  const isArabic = report.language === "arabic";

  const handleExport = () => {
    const content = JSON.stringify(report, null, 2);
    const blob = new Blob([content], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `audit-report-${report.audit_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5 max-w-3xl"
      dir={isArabic ? "rtl" : "ltr"}
    >
      {/* Report header */}
      <div className="bg-white border border-outline-variant rounded-xl p-5">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-primary/10">
              <span className="material-symbols-outlined text-[22px] text-primary">shield</span>
            </div>
            <div>
              <h2 className="font-bold text-[15px] text-foreground">{report.title}</h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Generated {formatDate(report.generated_at)} · {isArabic ? "Arabic" : "English"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Overall risk badge */}
            <span className={cn(
              "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-[12px] font-bold",
              risk.badge
            )}>
              <span className="material-symbols-outlined text-[14px]">{risk.icon}</span>
              {risk.label}
            </span>

            {/* Export */}
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-outline-variant rounded text-xs font-semibold text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
            >
              <span className="material-symbols-outlined text-[14px]">download</span>
              Export JSON
            </button>
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Critical", count: critical.length, badge: "bg-red-50 text-red-700 border-red-200", icon: "error" },
          { label: "Warnings", count: warnings.length, badge: "bg-amber-50 text-amber-700 border-amber-200", icon: "warning" },
          { label: "Informational", count: ok.length, badge: "bg-secondary text-muted-foreground border-outline-variant", icon: "info" },
        ].map(({ label, count, badge, icon }) => (
          <div key={label} className={cn("rounded-xl border p-4 text-center", badge)}>
            <span className="material-symbols-outlined text-[20px] mb-1">{icon}</span>
            <p className="text-2xl font-bold tabular-nums">{count}</p>
            <p className="text-[11px] font-semibold uppercase tracking-wide mt-0.5 opacity-80">{label}</p>
          </div>
        ))}
      </div>

      {/* Executive summary */}
      <div className="bg-white border border-outline-variant rounded-xl p-5">
        <h3 className="text-[13px] font-bold text-foreground mb-3 flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-primary">article</span>
          {isArabic ? "الملخص التنفيذي" : "Executive Summary"}
        </h3>
        <p
          className="text-[13px] text-on-surface-variant leading-relaxed"
          dir={isArabic ? "rtl" : "ltr"}
        >
          {report.executive_summary}
        </p>
      </div>

      {/* Documents reviewed */}
      <div className="bg-white border border-outline-variant rounded-xl p-5">
        <h3 className="text-[13px] font-bold text-foreground mb-3 flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-primary">folder_open</span>
          {isArabic ? "المستندات التي تمت مراجعتها" : "Documents Reviewed"}
        </h3>
        <div className="flex flex-wrap gap-2">
          {report.documents_reviewed.map((doc, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1.5 text-[12px] px-2.5 py-1 rounded-md bg-secondary border border-outline-variant text-muted-foreground"
            >
              <span className="material-symbols-outlined text-[13px]">description</span>
              {doc}
            </span>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-5">
          <h3 className="text-[13px] font-bold text-primary mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-[18px]">lightbulb</span>
            {isArabic ? "التوصيات" : "Recommendations"}
          </h3>
          <ol className="space-y-3">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-3 text-[13px] text-foreground">
                <span className="w-6 h-6 rounded-full bg-primary text-white text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <span dir={rec.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"} className="leading-relaxed">{rec}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Audit ID footer */}
      <div className="flex items-center justify-between text-xs text-muted-foreground border-t border-outline-variant pt-4">
        <span>
          Audit ID: <code className="font-mono bg-secondary border border-outline-variant px-1.5 py-0.5 rounded text-foreground">{report.audit_id.slice(0, 16)}…</code>
        </span>
        <span>{report.findings.length} finding{report.findings.length !== 1 ? "s" : ""} total</span>
      </div>
    </motion.div>
  );
}
