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

export function agentLabel(agent: string): string {
  const map: Record<string, string> = {
    extraction: "Document Extraction",
    planner: "Audit Planner",
    cross_checker: "Cross-Checker",
    report_writer: "Report Writer",
  };
  return map[agent] ?? agent;
}
