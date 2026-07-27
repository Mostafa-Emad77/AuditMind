"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import { listAudits } from "@/lib/api";
import { cn, formatDate } from "@/lib/utils";
import type { AuditSummary } from "@/types";

const RISK_BADGE: Record<string, string> = {
  critical: "bg-red-50 text-red-700 border-red-200",
  high: "bg-orange-50 text-orange-700 border-orange-200",
  medium: "bg-amber-50 text-amber-700 border-amber-200",
  low: "bg-blue-50 text-blue-700 border-blue-100",
  clean: "bg-emerald-50 text-emerald-700 border-emerald-200",
};

const STATUS_BADGE: Record<string, string> = {
  completed: "bg-emerald-50 text-emerald-700 border-emerald-200",
  processing: "bg-blue-50 text-blue-700 border-blue-100",
  pending: "bg-muted text-muted-foreground border-outline-variant",
  failed: "bg-red-50 text-red-700 border-red-200",
};

function AuditRow({ audit }: { audit: AuditSummary }) {
  const riskBadge = audit.overall_risk ? RISK_BADGE[audit.overall_risk] : null;

  return (
    <Link
      href={`/audit/${audit.audit_id}`}
      className="flex items-center justify-between gap-4 p-4 bg-white border border-outline-variant rounded-lg hover:border-primary/40 hover:bg-secondary/40 transition-colors"
    >
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-foreground truncate">
          {audit.filenames.length > 0 ? audit.filenames.join(", ") : "Untitled audit"}
        </p>
        <p className="text-xs text-muted-foreground mt-0.5">
          {formatDate(audit.created_at)} · {audit.document_count} document{audit.document_count === 1 ? "" : "s"}
        </p>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {audit.critical_count > 0 && (
          <span className="inline-flex items-center gap-1 bg-red-50 text-red-700 border border-red-200 px-2 py-0.5 rounded text-[11px] font-semibold">
            {audit.critical_count} Critical
          </span>
        )}
        {audit.warning_count > 0 && (
          <span className="inline-flex items-center gap-1 bg-amber-50 text-amber-700 border border-amber-200 px-2 py-0.5 rounded text-[11px] font-semibold">
            {audit.warning_count} Warning{audit.warning_count > 1 ? "s" : ""}
          </span>
        )}
        {riskBadge && (
          <span className={cn("inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold border", riskBadge)}>
            {audit.overall_risk?.toUpperCase()}
          </span>
        )}
        <span className={cn("inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold border capitalize", STATUS_BADGE[audit.status])}>
          {audit.status}
        </span>
        <span className="material-symbols-outlined text-[18px] text-muted-foreground">chevron_right</span>
      </div>
    </Link>
  );
}

export default function ArchivePage() {
  const [audits, setAudits] = useState<AuditSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listAudits()
      .then(setAudits)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Failed to load audit history"))
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top navigation */}
      <nav className="bg-white border-b border-outline-variant px-6 py-3">
        <div className="flex items-center justify-between w-full max-w-[1440px] mx-auto">
          <Link href="/" className="flex items-center gap-2 text-xl font-bold tracking-tight text-foreground">
            <Image src="/logo.png" alt="AuditMind Logo" width={32} height={32} className="object-contain" />
            AuditMind
          </Link>
          <Link
            href="/"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            New Audit
          </Link>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 px-6 py-10 max-w-[1440px] mx-auto w-full">
        <div className="max-w-3xl mx-auto space-y-6">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">Archive</h1>
            <p className="text-sm text-muted-foreground mt-1">Past audits, newest first.</p>
          </div>

          {isLoading && (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-16 bg-white border border-outline-variant rounded-lg shimmer" />
              ))}
            </div>
          )}

          {!isLoading && error && (
            <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">{error}</div>
          )}

          {!isLoading && !error && audits.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-center bg-white border border-outline-variant rounded-lg">
              <span className="material-symbols-outlined text-[40px] text-muted-foreground/40 mb-3">inventory_2</span>
              <p className="text-sm text-muted-foreground">No audits yet</p>
              <Link href="/" className="text-sm text-primary font-medium mt-2 hover:underline">
                Start your first audit
              </Link>
            </div>
          )}

          {!isLoading && !error && audits.length > 0 && (
            <div className="space-y-2">
              {audits.map((audit) => (
                <AuditRow key={audit.audit_id} audit={audit} />
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Material Symbols font */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');`}</style>
    </div>
  );
}
