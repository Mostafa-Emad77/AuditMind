"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createAuditEventSource, getAuditReport, getAuditStatus } from "@/lib/api";
import type {
  AgentName,
  AuditReport,
  AuditStreamState,
  Finding,
  ReasoningStep,
  SessionDocumentRef,
  SSEEvent,
} from "@/types";

const AGENT_ORDER: AgentName[] = [
  "extraction",
  "planner",
  "cross_checker",
  "report_writer",
];

const initialAgentStatuses = (): Record<
  AgentName,
  "pending" | "active" | "done"
> => ({
  extraction: "pending",
  planner: "pending",
  cross_checker: "pending",
  report_writer: "pending",
});

export function useAuditStream(auditId: string | null): AuditStreamState {
  const [steps, setSteps] = useState<ReasoningStep[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [report, setReport] = useState<AuditReport | null>(null);
  const [sessionDocuments, setSessionDocuments] = useState<SessionDocumentRef[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentStatuses, setAgentStatuses] = useState(initialAgentStatuses());

  const esRef = useRef<EventSource | null>(null);
  const activeAgentRef = useRef<AgentName | null>(null);

  const markAgentActive = useCallback((agent: AgentName) => {
    if (activeAgentRef.current === agent) return;
    activeAgentRef.current = agent;
    setAgentStatuses((prev) => {
      const next = { ...prev };
      // Mark all previous agents done
      const idx = AGENT_ORDER.indexOf(agent);
      AGENT_ORDER.forEach((a, i) => {
        if (i < idx) next[a] = "done";
      });
      next[agent] = "active";
      return next;
    });
  }, []);

  const markAgentDone = useCallback((agent: AgentName) => {
    setAgentStatuses((prev) => ({ ...prev, [agent]: "done" }));
  }, []);

  useEffect(() => {
    if (!auditId) return;

    // Clean up any existing connection
    if (esRef.current) {
      esRef.current.close();
    }

    setSteps([]);
    setFindings([]);
    setReport(null);
    setSessionDocuments([]);
    setIsRunning(true);
    setIsComplete(false);
    setError(null);
    setAgentStatuses(initialAgentStatuses());

    getAuditStatus(auditId)
      .then((s: { documents?: SessionDocumentRef[] }) => {
        setSessionDocuments(s.documents ?? []);
      })
      .catch(() => {});

    const es = createAuditEventSource(auditId);
    esRef.current = es;

    es.onmessage = (event) => {
      try {
        const data: SSEEvent = JSON.parse(event.data);

        switch (data.type) {
          case "connected":
            // Session connected, audit will begin
            break;

          case "reasoning_step": {
            const step = data.step;
            if (!step) break;

            markAgentActive(step.agent);

            // If this is a summary step, mark agent done
            if (step.step_type === "summary") {
              markAgentDone(step.agent);
            }

            setSteps((prev) => [...prev, step]);

            // Extract findings from finding steps
            if (step.step_type === "finding") {
              // Findings come via report_ready, but we can preview from steps
            }
            break;
          }

          case "report_ready": {
            // Fetch full report from API
            getAuditReport(auditId)
              .then((r) => {
                setReport(r);
                setFindings(r.findings);
              })
              .catch(console.error);
            break;
          }

          case "complete": {
            setIsRunning(false);
            setIsComplete(true);
            setAgentStatuses((prev) => {
              const next = { ...prev };
              AGENT_ORDER.forEach((a) => {
                if (next[a] !== "pending") next[a] = "done";
              });
              return next;
            });
            es.close();
            break;
          }

          case "error": {
            setError(data.message || "An error occurred during audit");
            setIsRunning(false);
            es.close();
            break;
          }
        }
      } catch (e) {
        console.error("Failed to parse SSE event:", e);
      }
    };

    es.onerror = () => {
      setError("Connection to audit stream lost. Please try again.");
      setIsRunning(false);
      es.close();
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [auditId, markAgentActive, markAgentDone]);

  return {
    steps,
    findings,
    report,
    sessionDocuments,
    isRunning,
    isComplete,
    error,
    agentStatuses,
  };
}
