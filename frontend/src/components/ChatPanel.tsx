"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Send, Sparkles } from "lucide-react";
import { streamChat, getChatHistory } from "@/lib/api";
import type { ChatMessage, ChatSource } from "@/types";
import { cn } from "@/lib/utils";

const isRTL = (t: string) => /[؀-ۿ]/.test(t);

const SUGGESTIONS = [
  "Why was the most critical finding flagged?",
  "What is the total amount paid according to the bank statements?",
  "Do the invoice totals match the contract value?",
];

function SourceChips({ sources }: { sources: ChatSource[] }) {
  if (!sources?.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {sources.slice(0, 6).map((s, i) => (
        <span
          key={i}
          title={s.text}
          className="inline-flex items-center gap-1 px-2 py-0.5 bg-muted border border-outline-variant rounded text-[10px] text-muted-foreground"
        >
          <span className="material-symbols-outlined text-[11px]">description</span>
          {s.doc_name}
          {s.page ? ` · p.${s.page}` : ""}
        </span>
      ))}
    </div>
  );
}

export function ChatPanel({ auditId, disabled }: { auditId: string; disabled?: boolean }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [draftAnswer, setDraftAnswer] = useState("");
  const [draftSources, setDraftSources] = useState<ChatSource[]>([]);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getChatHistory(auditId)
      .then(setMessages)
      .catch(() => {});
  }, [auditId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, draftAnswer]);

  const send = async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || streaming || disabled) return;
    setError(null);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setStreaming(true);
    setDraftAnswer("");
    setDraftSources([]);

    let acc = "";
    let srcs: ChatSource[] = [];
    try {
      await streamChat(auditId, q, {
        onSources: (s) => {
          srcs = s;
          setDraftSources(s);
        },
        onDelta: (t) => {
          acc += t;
          setDraftAnswer(acc);
        },
        onError: (msg) => setError(msg),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Chat request failed");
    } finally {
      if (acc.trim()) {
        setMessages((m) => [...m, { role: "assistant", content: acc, sources: srcs }]);
      }
      setDraftAnswer("");
      setDraftSources([]);
      setStreaming(false);
    }
  };

  const isEmpty = messages.length === 0 && !streaming;

  return (
    <div className="flex flex-col h-[calc(100vh-260px)] min-h-[420px] bg-white border border-outline-variant rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-5 py-3 border-b border-outline-variant bg-secondary">
        <Sparkles className="h-4 w-4 text-primary" />
        <p className="text-sm font-semibold text-foreground">Ask about these documents</p>
        <span className="text-[11px] text-muted-foreground ml-auto">
          Grounded in the audited files &amp; findings
        </span>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 space-y-4">
        {disabled && (
          <div className="text-center text-sm text-muted-foreground py-10">
            Q&amp;A becomes available once the audit completes.
          </div>
        )}

        {!disabled && isEmpty && (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Ask a question about the uploaded documents, the findings, or the reconciliation.
            </p>
            <div className="flex flex-col gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-left text-[13px] px-3 py-2 rounded-lg border border-outline-variant bg-secondary hover:bg-muted transition-colors text-foreground"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
            <div
              className={cn(
                "max-w-[80%] rounded-2xl px-4 py-2.5 text-[13px] leading-relaxed",
                m.role === "user"
                  ? "bg-primary text-white rounded-br-sm"
                  : "bg-secondary text-foreground border border-outline-variant rounded-bl-sm"
              )}
            >
              <p className="whitespace-pre-wrap" dir={isRTL(m.content) ? "rtl" : "ltr"}>
                {m.content}
              </p>
              {m.role === "assistant" && <SourceChips sources={m.sources ?? []} />}
            </div>
          </div>
        ))}

        {streaming && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-2xl rounded-bl-sm px-4 py-2.5 text-[13px] leading-relaxed bg-secondary text-foreground border border-outline-variant">
              {draftAnswer ? (
                <p className="whitespace-pre-wrap" dir={isRTL(draftAnswer) ? "rtl" : "ltr"}>
                  {draftAnswer}
                </p>
              ) : (
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Searching the documents…
                </span>
              )}
              <SourceChips sources={draftSources} />
            </div>
          </div>
        )}

        {error && (
          <div className="text-[12px] text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2">
            {error}
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t border-outline-variant p-3 bg-white">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            disabled={disabled || streaming}
            rows={1}
            placeholder={disabled ? "Available after the audit completes" : "Ask a question…"}
            dir={isRTL(input) ? "rtl" : "ltr"}
            className="flex-1 resize-none max-h-32 px-3 py-2 text-[13px] rounded-lg border border-outline-variant bg-secondary focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-60"
          />
          <button
            onClick={() => send()}
            disabled={disabled || streaming || !input.trim()}
            className="flex items-center justify-center h-9 w-9 rounded-lg bg-primary text-white disabled:opacity-40 hover:bg-[#003ea8] transition-colors shrink-0"
          >
            {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}
