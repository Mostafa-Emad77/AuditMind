"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Wrench,
  CheckCircle2,
  AlertTriangle,
  FileText,
  ChevronRight,
  ChevronDown,
  Loader2,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { agentLabel, severityBg } from "@/lib/utils";
import type { AgentName, ReasoningStep } from "@/types";

const AGENT_ORDER: AgentName[] = [
  "extraction",
  "planner",
  "cross_checker",
  "report_writer",
];

const AGENT_COLORS: Record<AgentName, string> = {
  extraction: "text-blue-400 bg-blue-400/10 border-blue-400/30",
  planner: "text-purple-400 bg-purple-400/10 border-purple-400/30",
  cross_checker: "text-amber-400 bg-amber-400/10 border-amber-400/30",
  report_writer: "text-emerald-400 bg-emerald-400/10 border-emerald-400/30",
};

const AGENT_ICONS: Record<AgentName, string> = {
  extraction: "📄",
  planner: "📋",
  cross_checker: "🔍",
  report_writer: "📊",
};

interface AgentStatusBadgeProps {
  agent: AgentName;
  status: "pending" | "active" | "done";
}

function AgentStatusBadge({ agent, status }: AgentStatusBadgeProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-all duration-300",
        status === "active" && AGENT_COLORS[agent],
        status === "done" && "text-emerald-400 bg-emerald-400/10 border-emerald-400/30",
        status === "pending" && "text-muted-foreground bg-secondary border-border opacity-50"
      )}
    >
      <span>{AGENT_ICONS[agent]}</span>
      <span>{agentLabel(agent)}</span>
      {status === "active" && (
        <Loader2 className="h-3 w-3 animate-spin" />
      )}
      {status === "done" && (
        <CheckCircle2 className="h-3 w-3" />
      )}
    </div>
  );
}

interface StepCardProps {
  step: ReasoningStep;
  isNew?: boolean;
}

function StepCard({ step, isNew }: StepCardProps) {
  const icons: Record<string, React.ReactNode> = {
    thought: <Brain className="h-3.5 w-3.5 text-muted-foreground mt-0.5 shrink-0" />,
    tool_call: <Wrench className="h-3.5 w-3.5 text-blue-400 mt-0.5 shrink-0" />,
    tool_result: <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 mt-0.5 shrink-0" />,
    finding: <AlertTriangle className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />,
    summary: <Sparkles className="h-3.5 w-3.5 text-primary mt-0.5 shrink-0" />,
  };

  const isToolCall = step.step_type === "tool_call";
  const isToolResult = step.step_type === "tool_result";
  const isFinding = step.step_type === "finding";
  const isSummary = step.step_type === "summary";

  return (
    <motion.div
      initial={isNew ? { opacity: 0, y: 8 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={cn(
        "flex gap-2.5 p-2.5 rounded-lg text-xs",
        isFinding && "bg-amber-400/5 border border-amber-400/20",
        isSummary && "bg-primary/5 border border-primary/20",
        isToolCall && "bg-blue-400/5 border border-blue-400/10",
        !isFinding && !isSummary && !isToolCall && "hover:bg-accent/30"
      )}
    >
      {icons[step.step_type] ?? <FileText className="h-3.5 w-3.5 text-muted-foreground mt-0.5 shrink-0" />}
      <div className="flex-1 min-w-0">
        {/* Agent badge */}
        <span
          className={cn(
            "inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded mb-1.5",
            AGENT_COLORS[step.agent]
          )}
        >
          {AGENT_ICONS[step.agent]} {agentLabel(step.agent)}
        </span>

        {/* Content */}
        <p
          className={cn(
            "leading-relaxed whitespace-pre-wrap break-words",
            isFinding ? "text-amber-300" : "text-foreground/80",
            isSummary && "font-medium text-foreground"
          )}
          dir={step.content.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
        >
          {step.content}
        </p>

        {/* Tool call details */}
        {isToolCall && step.tool_name && (
          <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-blue-400/70">
            <ChevronRight className="h-3 w-3" />
            <code className="font-mono">{step.tool_name}</code>
            {step.tool_input && (
              <span className="text-muted-foreground">
                ({Object.entries(step.tool_input).map(([k, v]) => `${k}: ${String(v).slice(0, 30)}`).join(", ")})
              </span>
            )}
          </div>
        )}

        {/* Tool result */}
        {isToolResult && step.tool_output && (
          <div className="mt-1.5 text-[10px] text-emerald-400/70 font-mono">
            → {step.tool_output}
          </div>
        )}
      </div>
    </motion.div>
  );
}

interface ReasoningPanelProps {
  steps: ReasoningStep[];
  agentStatuses: Record<AgentName, "pending" | "active" | "done">;
  isRunning: boolean;
  isComplete: boolean;
}

export function ReasoningPanel({
  steps,
  agentStatuses,
  isRunning,
  isComplete,
}: ReasoningPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const prevLengthRef = useRef(0);
  const [isCollapsed, setIsCollapsed] = useState(true);

  const visibleSteps = useMemo(() => {
    if (!isCollapsed) return steps;
    return steps.slice(-3);
  }, [isCollapsed, steps]);

  // Auto-scroll to bottom when new steps arrive
  useEffect(() => {
    if (steps.length > prevLengthRef.current) {
      prevLengthRef.current = steps.length;
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [steps.length]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold">Reasoning Trace</h2>
          </div>
          <div className="flex items-center gap-2">
            {isRunning && (
              <div className="flex items-center gap-1.5 text-xs text-primary">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>Thinking...</span>
              </div>
            )}
            {isComplete && (
              <div className="flex items-center gap-1.5 text-xs text-emerald-400">
                <CheckCircle2 className="h-3 w-3" />
                <span>Complete</span>
              </div>
            )}
            <button
              onClick={() => setIsCollapsed((prev) => !prev)}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded-md border border-border bg-secondary hover:bg-accent transition-colors"
            >
              {isCollapsed ? (
                <>
                  <ChevronRight className="h-3 w-3" />
                  Expand
                </>
              ) : (
                <>
                  <ChevronDown className="h-3 w-3" />
                  Collapse
                </>
              )}
            </button>
          </div>
        </div>

        {/* Agent pipeline status */}
        <div className="flex flex-wrap gap-1.5">
          {AGENT_ORDER.map((agent) => (
            <AgentStatusBadge
              key={agent}
              agent={agent}
              status={agentStatuses[agent]}
            />
          ))}
        </div>
      </div>

      {/* Steps feed */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
        {isCollapsed && steps.length > 3 && (
          <div className="text-[11px] px-2 py-1.5 rounded-md border border-border bg-secondary text-muted-foreground">
            Showing latest 3 of {steps.length} events.
          </div>
        )}
        {steps.length === 0 && !isRunning && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <Brain className="h-8 w-8 text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">
              Agent reasoning will appear here as the audit runs
            </p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              Upload documents and start an audit to see the live chain-of-thought
            </p>
          </div>
        )}

        <AnimatePresence mode="sync">
          {visibleSteps.map((step, i) => (
            <StepCard
              key={step.step_id}
              step={step}
              isNew={i === visibleSteps.length - 1 && isRunning}
            />
          ))}
        </AnimatePresence>

        {/* Typing indicator */}
        {isRunning && steps.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-1.5 px-2.5 py-2"
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

        <div ref={bottomRef} />
      </div>

      {/* Footer stats */}
      {steps.length > 0 && (
        <div className="px-4 py-2 border-t border-border">
          <p className="text-xs text-muted-foreground">
            {steps.length} reasoning step{steps.length !== 1 ? "s" : ""}
            {" · "}
            {steps.filter((s) => s.step_type === "tool_call").length} tool call{steps.filter((s) => s.step_type === "tool_call").length !== 1 ? "s" : ""}
            {" · "}
            {steps.filter((s) => s.step_type === "finding").length} finding{steps.filter((s) => s.step_type === "finding").length !== 1 ? "s" : ""}
          </p>
        </div>
      )}
    </div>
  );
}
