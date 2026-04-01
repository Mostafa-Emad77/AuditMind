"use client";

import { motion } from "framer-motion";
import {
  FileText,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Download,
  Shield,
} from "lucide-react";
import { cn, severityBg, formatDate } from "@/lib/utils";
import type { AuditReport as AuditReportType } from "@/types";

const RISK_CONFIG = {
  critical: { icon: AlertCircle, color: "text-red-400", bg: "bg-red-400/10 border-red-400/30", label: "CRITICAL RISK" },
  high: { icon: AlertTriangle, color: "text-orange-400", bg: "bg-orange-400/10 border-orange-400/30", label: "HIGH RISK" },
  medium: { icon: AlertTriangle, color: "text-yellow-400", bg: "bg-yellow-400/10 border-yellow-400/30", label: "MEDIUM RISK" },
  low: { icon: AlertTriangle, color: "text-blue-400", bg: "bg-blue-400/10 border-blue-400/30", label: "LOW RISK" },
  clean: { icon: CheckCircle2, color: "text-emerald-400", bg: "bg-emerald-400/10 border-emerald-400/30", label: "CLEAN" },
};

interface AuditReportProps {
  report: AuditReportType;
}

export function AuditReport({ report }: AuditReportProps) {
  const risk = RISK_CONFIG[report.overall_risk] ?? RISK_CONFIG.medium;
  const RiskIcon = risk.icon;

  const critical = report.findings.filter((f) => f.severity === "critical");
  const warnings = report.findings.filter((f) => f.severity === "warning");
  const ok = report.findings.filter((f) => f.severity === "ok");

  const isArabic = report.language === "arabic";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
      dir={isArabic ? "rtl" : "ltr"}
    >
      {/* Report header */}
      <div className="p-4 rounded-xl bg-card border border-border">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <Shield className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h2 className="font-semibold text-base">{report.title}</h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Generated {formatDate(report.generated_at)} · {report.language === "arabic" ? "Arabic" : "English"}
              </p>
            </div>
          </div>

          {/* Overall risk badge */}
          <div className={cn("flex items-center gap-2 px-4 py-2 rounded-lg border font-semibold text-sm", risk.bg)}>
            <RiskIcon className={cn("h-4 w-4", risk.color)} />
            <span className={risk.color}>{risk.label}</span>
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Critical", count: critical.length, color: "text-red-400", bg: "bg-red-400/5 border-red-400/20" },
          { label: "Warnings", count: warnings.length, color: "text-amber-400", bg: "bg-amber-400/5 border-amber-400/20" },
          { label: "OK", count: ok.length, color: "text-emerald-400", bg: "bg-emerald-400/5 border-emerald-400/20" },
        ].map(({ label, count, color, bg }) => (
          <div key={label} className={cn("rounded-lg border p-3 text-center", bg)}>
            <p className={cn("text-2xl font-bold", color)}>{count}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
          </div>
        ))}
      </div>

      {/* Executive summary */}
      <div className="p-4 rounded-xl bg-card border border-border">
        <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
          <FileText className="h-4 w-4 text-primary" />
          {isArabic ? "الملخص التنفيذي" : "Executive Summary"}
        </h3>
        <p
          className="text-sm text-foreground/80 leading-relaxed"
          dir={isArabic ? "rtl" : "ltr"}
        >
          {report.executive_summary}
        </p>
      </div>

      {/* Documents reviewed */}
      <div className="p-4 rounded-xl bg-card border border-border">
        <h3 className="text-sm font-semibold mb-2">
          {isArabic ? "المستندات التي تمت مراجعتها" : "Documents Reviewed"}
        </h3>
        <div className="flex flex-wrap gap-2">
          {report.documents_reviewed.map((doc, i) => (
            <span
              key={i}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md bg-secondary border border-border text-muted-foreground"
            >
              <FileText className="h-3 w-3" />
              {doc}
            </span>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div className="p-4 rounded-xl bg-primary/5 border border-primary/20">
          <h3 className="text-sm font-semibold mb-3 text-primary">
            {isArabic ? "التوصيات" : "Recommendations"}
          </h3>
          <ul className="space-y-2">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2.5 text-sm text-foreground/80">
                <span className="text-primary font-bold shrink-0 mt-0.5">{i + 1}.</span>
                <span dir={rec.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Download hint */}
      <div className="flex items-center justify-between text-xs text-muted-foreground px-1">
        <span>Audit ID: <code className="font-mono">{report.audit_id.slice(0, 16)}...</code></span>
        <button
          onClick={() => {
            const content = JSON.stringify(report, null, 2);
            const blob = new Blob([content], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `audit-report-${report.audit_id.slice(0, 8)}.json`;
            a.click();
          }}
          className="flex items-center gap-1.5 hover:text-primary transition-colors"
        >
          <Download className="h-3 w-3" />
          Export JSON
        </button>
      </div>
    </motion.div>
  );
}
