import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function severityColor(severity: string): string {
  const map: Record<string, string> = {
    critical: "text-red-400",
    warning: "text-amber-400",
    ok: "text-emerald-400",
    high: "text-orange-400",
    medium: "text-yellow-400",
    low: "text-blue-400",
    clean: "text-emerald-400",
  };
  return map[severity] ?? "text-muted-foreground";
}

export function severityBg(severity: string): string {
  const map: Record<string, string> = {
    critical: "bg-red-400/10 border-red-400/30 text-red-400",
    warning: "bg-amber-400/10 border-amber-400/30 text-amber-400",
    ok: "bg-emerald-400/10 border-emerald-400/30 text-emerald-400",
    high: "bg-orange-400/10 border-orange-400/30 text-orange-400",
    medium: "bg-yellow-400/10 border-yellow-400/30 text-yellow-400",
    low: "bg-blue-400/10 border-blue-400/30 text-blue-400",
    clean: "bg-emerald-400/10 border-emerald-400/30 text-emerald-400",
  };
  return map[severity] ?? "bg-muted border-border text-muted-foreground";
}

export function agentLabel(agent: string): string {
  const map: Record<string, string> = {
    extraction: "Document Extraction",
    planner: "Audit Planner",
    cross_checker: "Cross-Checker",
    report_writer: "Report Writer",
  };
  return map[agent] ?? agent;
}

export function agentIcon(agent: string): string {
  const map: Record<string, string> = {
    extraction: "📄",
    planner: "📋",
    cross_checker: "🔍",
    report_writer: "📊",
  };
  return map[agent] ?? "🤖";
}

export function docTypeLabel(type: string): string {
  const map: Record<string, string> = {
    invoice: "Invoice",
    contract: "Contract",
    balance_sheet: "Balance Sheet",
    bank_statement: "Bank Statement",
    audit_report: "Audit Report",
    unknown: "Unknown",
  };
  return map[type] ?? type;
}
