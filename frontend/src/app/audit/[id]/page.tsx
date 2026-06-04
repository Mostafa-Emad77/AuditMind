"use client";

import Image from "next/image";
import { use, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { AlertCircle, AlertTriangle, FileText, Loader2 } from "lucide-react";
import { useAuditStream } from "@/hooks/useAuditStream";
import { ReasoningPanel } from "@/components/ReasoningPanel";
import { FindingsTable } from "@/components/FindingsTable";
import { AuditReport } from "@/components/AuditReport";
import { ReconciliationPanel } from "@/components/ReconciliationPanel";
import { FindingDetailSheet } from "@/components/FindingDetailSheet";
import { cn } from "@/lib/utils";
import type { Finding } from "@/types";

type TabId = "findings" | "reconciliation" | "report";

const TABS: { id: TabId; label: string }[] = [
  { id: "findings", label: "Findings" },
  { id: "reconciliation", label: "Reconciliation" },
  { id: "report", label: "Report Preview" },
];

export default function AuditPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: auditId } = use(params);
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabId>("findings");
  const [detailFinding, setDetailFinding] = useState<Finding | null>(null);

  const { steps, findings, report, sessionDocuments, isRunning, isComplete, error, agentStatuses } =
    useAuditStream(auditId);

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
          (findings.reduce((sum, f) => sum + f.confidence_score, 0) / findings.length) * 100
        )
      : 0;

  const hasLikelyNonFinancialDocs = (report?.documents_reviewed ?? []).some((doc) =>
    /(cv|resume|profile)/i.test(doc)
  );
  const showDocWarning = hasLikelyNonFinancialDocs;
  const showSingleDocNotice = isComplete && (report?.documents_reviewed?.length ?? 0) === 1;

  /* Progress percentage based on agent states */
  const progressPct = (() => {
    const order = ["extraction", "planner", "cross_checker", "report_writer"] as const;
    const doneCount = order.filter((a) => agentStatuses[a] === "done").length;
    const activeCount = order.filter((a) => agentStatuses[a] === "active").length;
    return Math.round(((doneCount + activeCount * 0.5) / order.length) * 100);
  })();

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top navigation */}
      <header className="bg-white border-b border-outline-variant flex justify-between items-center px-6 py-3 w-full z-50">
        <div className="flex items-center gap-6">
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-2 text-xl font-bold tracking-tight text-foreground"
          >
            <Image src="/logo.png" alt="AuditMind Logo" width={28} height={28} className="object-contain" />
            AuditMind
          </button>
          <nav className="hidden md:flex gap-1">
            <a href="/" className="text-muted-foreground hover:text-foreground text-sm px-2 py-1 rounded hover:bg-secondary transition-colors">
              Dashboard
            </a>
            <a href="#" className="text-primary border-b-2 border-primary text-sm font-semibold px-2 pb-0.5">
              Workpapers
            </a>
            {["Analytics", "Archive"].map((l) => (
              <a key={l} href="#" className="text-muted-foreground hover:text-foreground text-sm px-2 py-1 rounded hover:bg-secondary transition-colors">
                {l}
              </a>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-2">
          {criticalCount > 0 && (
            <span className="inline-flex items-center gap-1 bg-red-50 text-red-700 border border-red-200 px-2.5 py-1 rounded text-xs font-semibold">
              <span className="material-symbols-outlined text-[14px]">error</span>
              {criticalCount} Critical
            </span>
          )}
          {warningCount > 0 && (
            <span className="inline-flex items-center gap-1 bg-amber-50 text-amber-700 border border-amber-200 px-2.5 py-1 rounded text-xs font-semibold">
              <span className="material-symbols-outlined text-[14px]">warning</span>
              {warningCount} Warning{warningCount > 1 ? "s" : ""}
            </span>
          )}
          {isRunning && (
            <div className="flex items-center gap-1.5 text-xs text-primary">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Analyzing...
            </div>
          )}
          {isComplete && !error && (
            <span className="text-xs text-emerald-700 font-medium flex items-center gap-1">
              <span className="material-symbols-outlined text-[14px]">check_circle</span>
              Audit complete
            </span>
          )}
          {["language", "settings", "notifications"].map((icon) => (
            <button key={icon} className="p-2 hover:bg-secondary rounded-full transition-colors text-muted-foreground">
              <span className="material-symbols-outlined text-[20px]">{icon}</span>
            </button>
          ))}
          <div className="h-8 w-8 bg-muted rounded-full flex items-center justify-center border border-outline-variant">
            <span className="material-symbols-outlined text-muted-foreground text-[18px]">person</span>
          </div>
        </div>
      </header>

      {/* Body: sidebar + content */}
      <div className="flex flex-1 overflow-hidden" style={{ height: "calc(100vh - 57px)" }}>
        {/* Left sidebar — Reasoning Trace */}
        <aside className="bg-[#f8fafc] border-r border-outline-variant w-60 shrink-0 flex flex-col z-40 h-full">
          <ReasoningPanel
            steps={steps}
            agentStatuses={agentStatuses}
            isRunning={isRunning}
            isComplete={isComplete}
          />
        </aside>

        {/* Main canvas */}
        <main className="flex-1 flex flex-col bg-background overflow-hidden">
          {/* Status header */}
          <div className="px-8 py-5 border-b border-outline-variant bg-white shrink-0">
            {/* Banners */}
            {error && (
              <div className="mb-3 px-4 py-2 bg-red-50 border border-red-200 rounded flex items-center gap-2 text-red-700 text-sm">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {error}
              </div>
            )}
            {showDocWarning && (
              <div className="mb-3 px-4 py-2 bg-amber-50 border border-amber-200 rounded flex items-start gap-2 text-amber-700 text-sm">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium">Low relevance document type detected</p>
                  <p className="text-xs text-amber-600 mt-0.5">
                    Some uploaded files look non-financial. Findings may be limited.
                  </p>
                </div>
              </div>
            )}
            {showSingleDocNotice && (
              <div className="mb-3 px-4 py-2 bg-blue-50 border border-blue-200 rounded flex items-start gap-2 text-blue-700 text-sm">
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium">Single-document audit scope</p>
                  <p className="text-xs text-blue-600 mt-0.5">
                    Upload at least 2 related files for cross-document contradiction detection.
                  </p>
                </div>
              </div>
            )}

            <div className="flex justify-between items-start mb-4">
              <div>
                <h1 className="text-[28px] font-bold tracking-tight text-foreground flex items-center gap-3">
                  {isComplete ? "Audit Complete" : isRunning ? "Audit in Progress" : "Audit"}
                  {isRunning && (
                    <span className="text-primary text-[22px] font-semibold">({progressPct}%)</span>
                  )}
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                  Audit ID: <code className="font-mono text-xs bg-secondary px-1.5 py-0.5 rounded">{auditId.slice(0, 16)}...</code>
                </p>
              </div>
              {/* Severity counters */}
              <div className="flex gap-2">
                <div className="flex items-center gap-1.5 bg-red-50 text-red-700 border border-red-200 px-3 py-1.5 rounded text-xs font-semibold">
                  <span className="material-symbols-outlined text-[14px]">error</span>
                  Critical
                  <span className="font-bold ml-1">{criticalCount}</span>
                </div>
                <div className="flex items-center gap-1.5 bg-amber-50 text-amber-700 border border-amber-200 px-3 py-1.5 rounded text-xs font-semibold">
                  <span className="material-symbols-outlined text-[14px]">warning</span>
                  Warning
                  <span className="font-bold ml-1">{warningCount}</span>
                </div>
                <div className="flex items-center gap-1.5 bg-muted text-muted-foreground border border-outline-variant px-3 py-1.5 rounded text-xs font-semibold">
                  <span className="material-symbols-outlined text-[14px]">info</span>
                  Info
                  <span className="font-bold ml-1">{okCount}</span>
                </div>
              </div>
            </div>

            {/* Progress bar */}
            {isRunning && (
              <div className="w-full bg-muted h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-primary h-full rounded-full transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            )}
          </div>

          {/* Content tabs */}
          <div className="px-8 border-b border-outline-variant flex gap-6 shrink-0 bg-white">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "py-3.5 text-xs font-semibold uppercase tracking-widest transition-colors",
                  activeTab === tab.id
                    ? "text-primary border-b-2 border-primary"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {tab.label}
                {tab.id === "findings" && findings.length > 0 && (
                  <span className="ml-1.5 bg-primary text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">
                    {findings.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto p-8 bg-background">
            {activeTab === "findings" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} key="findings">
                {/* Stats row */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white rounded-lg border border-red-200 p-4">
                    <p className="text-xl font-bold text-red-600 tabular-nums">{criticalCount}</p>
                    <p className="text-xs text-muted-foreground mt-1">Critical</p>
                  </div>
                  <div className="bg-white rounded-lg border border-amber-200 p-4">
                    <p className="text-xl font-bold text-amber-600 tabular-nums">{warningCount}</p>
                    <p className="text-xs text-muted-foreground mt-1">Warnings</p>
                  </div>
                  <div className="bg-white rounded-lg border border-emerald-200 p-4">
                    <p className="text-xl font-bold text-emerald-600 tabular-nums">{okCount}</p>
                    <p className="text-xs text-muted-foreground mt-1">Informational</p>
                  </div>
                  <div className="bg-white rounded-lg border border-outline-variant p-4">
                    <p className="text-xl font-bold text-primary tabular-nums">{avgConfidence}%</p>
                    <p className="text-xs text-muted-foreground mt-1">Avg confidence</p>
                  </div>
                </div>

                <FindingsTable
                  findings={findings}
                  isLoading={isRunning && findings.length === 0}
                  onOpenDetail={(f) => setDetailFinding(f)}
                />
              </motion.div>
            )}

            {activeTab === "reconciliation" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} key="reconciliation">
                <ReconciliationPanel
                  report={report}
                  isComplete={isComplete}
                  isRunning={isRunning}
                  docFilenameById={docFilenameById}
                />
              </motion.div>
            )}

            {activeTab === "report" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} key="report">
                {report ? (
                  <AuditReport report={report} />
                ) : (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <FileText className="h-8 w-8 text-muted-foreground/40 mb-3" />
                    <p className="text-sm text-muted-foreground">
                      {isRunning
                        ? "Report will be generated when audit completes..."
                        : "No report available yet."}
                    </p>
                    {isRunning && <Loader2 className="h-4 w-4 animate-spin text-primary mt-3" />}
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </main>
      </div>

      <FindingDetailSheet
        finding={detailFinding}
        open={detailFinding !== null}
        onOpenChange={(open) => { if (!open) setDetailFinding(null); }}
        docFilenameById={docFilenameById}
      />

      {/* Material Symbols */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');`}</style>
    </div>
  );
}
