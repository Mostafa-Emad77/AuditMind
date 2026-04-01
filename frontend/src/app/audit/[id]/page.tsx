"use client";

import { use, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Brain,
  ArrowLeft,
  FileText,
  Scale,
  BarChart3,
  AlertCircle,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { useAuditStream } from "@/hooks/useAuditStream";
import { ReasoningPanel } from "@/components/ReasoningPanel";
import { FindingsTable } from "@/components/FindingsTable";
import { AuditReport } from "@/components/AuditReport";
import { ReconciliationPanel } from "@/components/ReconciliationPanel";
import { FindingDetailSheet } from "@/components/FindingDetailSheet";
import { cn, severityBg } from "@/lib/utils";
import type { Finding } from "@/types";

type TabId = "findings" | "reconciliation" | "report";

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: "findings", label: "Findings", icon: <AlertCircle className="h-3.5 w-3.5" /> },
  { id: "reconciliation", label: "Reconciliation", icon: <Scale className="h-3.5 w-3.5" /> },
  { id: "report", label: "Report", icon: <BarChart3 className="h-3.5 w-3.5" /> },
];

export default function AuditPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: auditId } = use(params);
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabId>("findings");
  const [detailFinding, setDetailFinding] = useState<Finding | null>(null);

  const {
    steps,
    findings,
    report,
    sessionDocuments,
    isRunning,
    isComplete,
    error,
    agentStatuses,
  } = useAuditStream(auditId);

  const docFilenameById = useMemo(
    () => Object.fromEntries(sessionDocuments.map((d) => [d.doc_id, d.filename])),
    [sessionDocuments]
  );

  const criticalCount = findings.filter((f) => f.severity === "critical").length;
  const warningCount = findings.filter((f) => f.severity === "warning").length;
  const okCount = findings.filter((f) => f.severity === "ok").length;
  const avgConfidence =
    findings.length > 0
      ? Math.round(
          (findings.reduce((sum, finding) => sum + finding.confidence_score, 0) /
            findings.length) *
            100
        )
      : 0;
  const hasLikelyNonFinancialDocs = (report?.documents_reviewed ?? []).some((doc) =>
    /(cv|resume|profile)/i.test(doc)
  );
  const documentCount = report?.documents_reviewed?.length ?? 0;
  const showDocWarning = hasLikelyNonFinancialDocs;
  const showSingleDocNotice = isComplete && documentCount === 1;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top navigation */}
      <nav className="border-b border-border px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </button>
          <div className="h-4 w-px bg-border" />
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            <span className="font-semibold text-sm">AuditMind</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <code className="text-xs text-muted-foreground font-mono bg-secondary px-2 py-0.5 rounded">
            {auditId.slice(0, 16)}...
          </code>
        </div>

        {/* Status indicators */}
        <div className="flex items-center gap-3">
          {criticalCount > 0 && (
            <span className={cn("text-xs px-2 py-0.5 rounded-full border font-medium", severityBg("critical"))}>
              {criticalCount} Critical
            </span>
          )}
          {warningCount > 0 && (
            <span className={cn("text-xs px-2 py-0.5 rounded-full border font-medium", severityBg("warning"))}>
              {warningCount} Warning{warningCount > 1 ? "s" : ""}
            </span>
          )}
          {isRunning && (
            <div className="flex items-center gap-1.5 text-xs text-primary">
              <Loader2 className="h-3 w-3 animate-spin" />
              Analyzing...
            </div>
          )}
          {isComplete && !error && (
            <span className="text-xs text-emerald-400">Audit complete</span>
          )}
        </div>
      </nav>

      {/* Main layout: left panel (60%) + right reasoning panel (40%) */}
      <div className="flex-1 flex overflow-hidden" style={{ height: "calc(100vh - 57px)" }}>
        {/* Left panel */}
        <div className="flex-1 flex flex-col overflow-hidden border-r border-border">
          {/* Error banner */}
          {error && (
            <div className="px-5 py-3 bg-destructive/10 border-b border-destructive/20 flex items-center gap-2 text-destructive text-sm">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
          {showDocWarning && (
            <div className="px-5 py-3 bg-amber-500/10 border-b border-amber-500/30 flex items-start gap-2 text-amber-200 text-sm">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Low relevance document type detected</p>
                <p className="text-xs text-amber-200/80 mt-0.5">
                  Some uploaded files look non-financial. Findings may be limited and a CLEAN result can be misleading.
                </p>
              </div>
            </div>
          )}
          {showSingleDocNotice && (
            <div className="px-5 py-3 bg-blue-500/10 border-b border-blue-500/30 flex items-start gap-2 text-blue-200 text-sm">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Single-document audit scope</p>
                <p className="text-xs text-blue-200/80 mt-0.5">
                  Cross-document contradiction detection is limited with one document. Upload at least 2 related files (e.g. invoice + contract + statement) for stronger checks.
                </p>
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="flex items-center gap-1 px-5 py-3 border-b border-border">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                  activeTab === tab.id
                    ? "bg-primary/10 text-primary border border-primary/20"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                )}
              >
                {tab.icon}
                {tab.label}
                {tab.id === "findings" && findings.length > 0 && (
                  <span className="ml-1 px-1.5 py-0.5 rounded-full bg-primary/20 text-primary text-[10px] font-bold">
                    {findings.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto p-5">
            {activeTab === "findings" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                key="findings"
              >
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                  <div className="rounded-lg border border-red-400/25 bg-red-400/5 p-3">
                    <p className="text-xl font-semibold text-red-400">{criticalCount}</p>
                    <p className="text-xs text-muted-foreground">Critical</p>
                  </div>
                  <div className="rounded-lg border border-amber-400/25 bg-amber-400/5 p-3">
                    <p className="text-xl font-semibold text-amber-400">{warningCount}</p>
                    <p className="text-xs text-muted-foreground">Warnings</p>
                  </div>
                  <div className="rounded-lg border border-emerald-400/25 bg-emerald-400/5 p-3">
                    <p className="text-xl font-semibold text-emerald-400">{okCount}</p>
                    <p className="text-xs text-muted-foreground">OK</p>
                  </div>
                  <div className="rounded-lg border border-primary/25 bg-primary/5 p-3">
                    <p className="text-xl font-semibold text-primary">{avgConfidence}%</p>
                    <p className="text-xs text-muted-foreground">Avg confidence</p>
                  </div>
                </div>

                <h2 className="text-sm font-semibold mb-4 flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-primary" />
                  Audit Findings
                  {isRunning && (
                    <span className="text-xs text-muted-foreground font-normal">(live)</span>
                  )}
                </h2>
                <FindingsTable
                  findings={findings}
                  isLoading={isRunning && findings.length === 0}
                  onOpenDetail={(f) => setDetailFinding(f)}
                />
              </motion.div>
            )}

            {activeTab === "reconciliation" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                key="reconciliation"
              >
                <ReconciliationPanel
                  report={report}
                  isComplete={isComplete}
                  isRunning={isRunning}
                  docFilenameById={docFilenameById}
                />
              </motion.div>
            )}

            {activeTab === "report" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                key="report"
              >
                {report ? (
                  <AuditReport report={report} />
                ) : (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <FileText className="h-8 w-8 text-muted-foreground/30 mb-3" />
                    <p className="text-sm text-muted-foreground">
                      {isRunning
                        ? "Report will be generated when audit completes..."
                        : "No report available yet. Start an audit first."}
                    </p>
                    {isRunning && (
                      <Loader2 className="h-4 w-4 animate-spin text-primary mt-3" />
                    )}
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>

        {/* Right: Reasoning panel */}
        <div className="w-[400px] shrink-0 flex flex-col overflow-hidden bg-card/30">
          <ReasoningPanel
            steps={steps}
            agentStatuses={agentStatuses}
            isRunning={isRunning}
            isComplete={isComplete}
          />
        </div>
      </div>

      <FindingDetailSheet
        finding={detailFinding}
        open={detailFinding !== null}
        onOpenChange={(open) => {
          if (!open) setDetailFinding(null);
        }}
        docFilenameById={docFilenameById}
      />
    </div>
  );
}
