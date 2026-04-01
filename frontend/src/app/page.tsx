"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, FileSearch, ChevronRight, ShieldAlert, Scale, CheckCircle2 } from "lucide-react";
import { FileUpload } from "@/components/FileUpload";
import type { UploadResponse } from "@/types";

export default function HomePage() {
  const [showHowItWorks, setShowHowItWorks] = useState(true);
  const GITHUB_URL = "https://github.com";

  const handleUploadComplete = (response: UploadResponse) => {
    // Navigate to audit page
    window.location.href = `/audit/${response.audit_id}`;
  };

  return (
    <div className="min-h-screen bg-background flex flex-col relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 opacity-20 [background-image:radial-gradient(circle_at_1px_1px,hsl(var(--border))_1px,transparent_0)] [background-size:28px_28px]" />
      {/* Top nav */}
      <nav className="border-b border-border px-6 py-4 relative z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <Brain className="h-5 w-5 text-primary" />
            </div>
            <span className="font-bold text-base tracking-tight">AuditMind</span>
          </div>
          <div className="hidden md:flex items-center gap-5 text-xs text-muted-foreground">
            <a href="#features" className="hover:text-foreground transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-foreground transition-colors">How it Works</a>
            <a href="#upload" className="hover:text-foreground transition-colors">Start Audit</a>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:text-primary/80 transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* Hero + upload */}
      <main className="flex-1 flex items-center justify-center px-6 py-12 relative z-10">
        <div className="w-full max-w-7xl grid grid-cols-1 xl:grid-cols-[300px_minmax(0,760px)_300px] gap-8 items-start">
          <aside className="hidden xl:block rounded-2xl border border-border bg-card/50 p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-amber-400" />
              Sample Finding
            </h3>
            <div className="rounded-lg border border-amber-400/30 bg-amber-400/5 p-3 space-y-2">
              <p className="text-xs font-medium text-amber-300">Amount mismatch detected</p>
              <p className="text-xs text-muted-foreground">
                Invoice shows <span className="text-foreground">50,000 EGP</span> while contract shows <span className="text-foreground">45,000 EGP</span>.
              </p>
              <p className="text-[11px] text-amber-200/80">Confidence: 92%</p>
            </div>
            <p className="text-[11px] text-muted-foreground mt-3">
              Real audits surface issues like this automatically.
            </p>
          </aside>

          <div className="w-full max-w-3xl mx-auto">
          {/* Hero */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="text-center mb-10"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-medium mb-6">
              <Brain className="h-3.5 w-3.5" />
              Powered by LangGraph + GraphRAG
            </div>
            <h1 className="text-4xl font-bold tracking-tight mb-4 bg-gradient-to-b from-foreground to-foreground/60 bg-clip-text text-transparent">
              Audit financial documents
              <br />
              <span className="text-primary">in Arabic and English</span>
            </h1>
            <p className="text-muted-foreground text-base max-w-lg mx-auto leading-relaxed">
              Upload invoices, contracts, bank statements, and balance sheets.
              Our AI agents will autonomously detect contradictions, verify consistency,
              and produce a bilingual audit report — with every reasoning step visible.
            </p>
          </motion.div>

          {/* Feature pills */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="space-y-3 mb-8"
            id="features"
          >
            <div className="flex items-center justify-center gap-2 flex-wrap">
              <span className="text-[11px] text-muted-foreground">Powered by:</span>
              {["LangGraph", "GraphRAG", "Neo4j", "Qdrant"].map((feat) => (
                <span
                  key={feat}
                  className="text-xs px-2.5 py-1 rounded-full bg-secondary border border-border text-muted-foreground"
                >
                  {feat}
                </span>
              ))}
            </div>
            <div className="flex items-center justify-center gap-2 flex-wrap">
              <span className="text-[11px] text-muted-foreground">Features:</span>
              {[
                "Cross-document contradiction detection",
                "Live chain-of-thought reasoning",
                "Arabic + English OCR",
                "Hybrid vector + graph search",
              ].map((feat) => (
                <span
                  key={feat}
                  className="text-xs px-2.5 py-1 rounded-full bg-secondary border border-border text-muted-foreground"
                >
                  {feat}
                </span>
              ))}
            </div>
          </motion.div>

          {/* Upload card */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.15 }}
            className="rounded-2xl border border-border bg-card p-6 shadow-2xl shadow-black/30"
            id="upload"
          >
            <div className="flex items-center gap-2 mb-5">
              <FileSearch className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">Upload Documents to Audit</h2>
            </div>
            <FileUpload onUploadComplete={handleUploadComplete} />
          </motion.div>

          {/* How it works */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.3 }}
            className="mt-10"
            id="how-it-works"
          >
            <button
              className="mx-auto block text-center text-xs text-muted-foreground mb-4 hover:text-foreground transition-colors"
              onClick={() => setShowHowItWorks((prev) => !prev)}
            >
              {showHowItWorks ? "Hide workflow" : "How it works"}
            </button>
            <AnimatePresence initial={false}>
            {showHowItWorks && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
              >
            <div className="flex items-center justify-center gap-0 flex-wrap">
              {[
                { icon: "📤", label: "Upload PDFs" },
                { icon: "📄", label: "OCR & Extract" },
                { icon: "🧠", label: "Build Graph" },
                { icon: "🔍", label: "Cross-Check" },
                { icon: "📊", label: "Audit Report" },
              ].map((step, i) => (
                <div key={step.label} className="flex items-center">
                  <div className="flex flex-col items-center gap-1 px-3 py-2">
                    <span className="text-lg">{step.icon}</span>
                    <span className="text-[10px] text-muted-foreground whitespace-nowrap">{step.label}</span>
                  </div>
                  {i < 4 && <ChevronRight className="h-3 w-3 text-muted-foreground/40 mx-0.5" />}
                </div>
              ))}
            </div>
              </motion.div>
            )}
            </AnimatePresence>
          </motion.div>
          </div>

          <aside className="hidden xl:block rounded-2xl border border-border bg-card/50 p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Scale className="h-4 w-4 text-primary" />
              Cross-document reconciliation
            </h3>
            <ul className="space-y-2.5 text-xs text-muted-foreground leading-relaxed">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                Automatically surfaces financial discrepancies across contracts, invoices, and statements
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                Flags entity conflicts including legal name mismatches and unauthorized payees
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                Every finding linked to exact source passages in original documents
              </li>
            </ul>
          </aside>
        </div>
      </main>
    </div>
  );
}
