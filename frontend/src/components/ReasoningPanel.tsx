"use client";

import { useEffect, useMemo, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, Loader2 } from "lucide-react";
import { cn, agentLabel } from "@/lib/utils";
import type { AgentName, ReasoningStep } from "@/types";

const AGENT_ORDER: AgentName[] = ["extraction", "planner", "cross_checker", "report_writer"];

const AGENT_ICONS: Record<AgentName, string> = {
  extraction: "quick_reference_all",
  planner: "account_tree",
  cross_checker: "rule",
  report_writer: "summarize",
};

interface ReasoningPanelProps {
  steps: ReasoningStep[];
  agentStatuses: Record<AgentName, "pending" | "active" | "done">;
  isRunning: boolean;
  isComplete: boolean;
}

export function ReasoningPanel({ steps, agentStatuses, isRunning, isComplete }: ReasoningPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const prevLengthRef = useRef(0);

  useEffect(() => {
    if (steps.length > prevLengthRef.current) {
      prevLengthRef.current = steps.length;
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [steps.length]);

  const visibleSteps = useMemo(() => steps.slice(-20), [steps]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 pt-6 pb-4 flex flex-col gap-1.5">
        <div className="flex items-center gap-2.5">
          {isRunning && (
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          )}
          {isComplete && (
            <div className="w-2 h-2 rounded-full bg-emerald-500" />
          )}
          {!isRunning && !isComplete && (
            <div className="w-2 h-2 rounded-full bg-outline-variant" />
          )}
          <span className="text-sm font-bold text-foreground">Reasoning Trace</span>
        </div>
        <span className="text-xs text-muted-foreground">AI Agent Activity</span>
      </div>

      {/* Agent pipeline nav */}
      <nav className="flex flex-col gap-0.5 flex-1 px-2 overflow-y-auto">
        {AGENT_ORDER.map((agent) => {
          const status = agentStatuses[agent];
          const isActive = status === "active";
          const isDone = status === "done";
          return (
            <div
              key={agent}
              className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-sm border-l-4 transition-all text-xs font-semibold uppercase tracking-wide",
                isActive &&
                  "bg-white border-primary text-primary shadow-sm",
                isDone &&
                  "border-transparent text-muted-foreground hover:bg-white hover:text-foreground",
                !isActive && !isDone &&
                  "border-transparent text-muted-foreground/50"
              )}
            >
              <span className="material-symbols-outlined text-[18px]">{AGENT_ICONS[agent]}</span>
              <span className="flex-1">{agentLabel(agent)}</span>
              {isActive && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {isDone && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />}
            </div>
          );
        })}

        {/* Step feed */}
        {visibleSteps.length > 0 && (
          <div className="mt-4 px-2 space-y-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2 px-2">
              Live Events
            </p>
            <AnimatePresence mode="sync">
              {visibleSteps.map((step, i) => (
                <motion.div
                  key={step.step_id}
                  initial={i === visibleSteps.length - 1 && isRunning ? { opacity: 0, y: 4 } : false}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={cn(
                    "flex gap-2 p-2 rounded text-[11px] leading-snug",
                    step.step_type === "finding" && "bg-amber-50 border border-amber-200 text-amber-800",
                    step.step_type === "tool_call" && "bg-blue-50 border border-blue-100 text-blue-800",
                    step.step_type === "summary" && "bg-secondary border border-outline-variant text-foreground font-medium",
                    !["finding", "tool_call", "summary"].includes(step.step_type) && "text-muted-foreground"
                  )}
                >
                  <span className="material-symbols-outlined text-[13px] mt-0.5 shrink-0">
                    {step.step_type === "thought" ? "psychology" :
                     step.step_type === "tool_call" ? "build" :
                     step.step_type === "tool_result" ? "check" :
                     step.step_type === "finding" ? "warning" :
                     step.step_type === "summary" ? "auto_awesome" : "description"}
                  </span>
                  <span
                    className="flex-1 min-w-0 break-words"
                    dir={step.content.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                  >
                    {step.content.slice(0, 120)}{step.content.length > 120 ? "..." : ""}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>

            {isRunning && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex gap-1.5 px-2 py-2"
              >
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="h-1.5 w-1.5 rounded-full bg-primary"
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ repeat: Infinity, duration: 1, delay: i * 0.2 }}
                  />
                ))}
              </motion.div>
            )}
          </div>
        )}

        <div ref={bottomRef} />
      </nav>
    </div>
  );
}
