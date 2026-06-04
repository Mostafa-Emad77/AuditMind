"use client";

import Image from "next/image";
import { FileUpload } from "@/components/FileUpload";
import type { UploadResponse } from "@/types";

export default function HomePage() {
  const handleUploadComplete = (response: UploadResponse) => {
    window.location.href = `/audit/${response.audit_id}`;
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top navigation */}
      <nav className="bg-white border-b border-outline-variant px-6 py-3">
        <div className="flex justify-between items-center w-full max-w-[1440px] mx-auto">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 text-xl font-bold tracking-tight text-foreground">
              <Image src="/logo.png" alt="AuditMind Logo" width={32} height={32} className="object-contain" />
              AuditMind
            </div>
            <div className="hidden md:flex gap-1 ml-4">
              <a href="#" className="text-primary border-b-2 border-primary pb-0.5 text-sm font-medium px-2 py-1">
                Dashboard
              </a>
              {["Workpapers", "Analytics", "Archive"].map((link) => (
                <a
                  key={link}
                  href="#"
                  className="text-muted-foreground hover:text-foreground text-sm transition-colors px-2 py-1 rounded hover:bg-secondary"
                >
                  {link}
                </a>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            {["language", "settings", "notifications"].map((icon) => (
              <button
                key={icon}
                className="p-2 hover:bg-secondary rounded-full transition-colors"
                aria-label={icon}
              >
                <span className="material-symbols-outlined text-[22px]">{icon}</span>
              </button>
            ))}
            <div className="ml-1 h-8 w-8 bg-muted rounded-full flex items-center justify-center border border-outline-variant overflow-hidden">
              <span className="material-symbols-outlined text-muted-foreground">person</span>
            </div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center pt-10 pb-12 px-6 max-w-[1440px] mx-auto w-full">
        {/* Hero */}
        <section className="text-center mb-10 max-w-3xl w-full">
          <h1 className="text-[32px] leading-10 font-bold tracking-tight text-foreground mb-3">
            AI-Powered Precision for Financial Audits
          </h1>
          <p className="text-base text-muted-foreground leading-6">
            Streamline your verification process with automated data extraction,
            cross-referencing, and anomaly detection. Upload your documents to begin.
          </p>
        </section>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-5 w-full max-w-5xl">
          {/* Upload zone — 8 cols */}
          <div className="col-span-1 md:col-span-8 bg-white border border-outline-variant rounded-xl p-8 flex flex-col justify-center min-h-[300px]">
            <FileUpload onUploadComplete={handleUploadComplete} />
          </div>

          {/* Config + action — 4 cols */}
          <div className="col-span-1 md:col-span-4 flex flex-col gap-5">
            {/* Audit configuration card */}
            <div className="bg-white border border-outline-variant rounded-xl p-4 flex flex-col gap-4">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[20px] text-muted-foreground">tune</span>
                <h3 className="text-base font-semibold text-foreground">Audit Configuration</h3>
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-[12px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Document Language
                </label>
                <div className="flex bg-accent rounded-lg p-1 gap-1">
                  <button className="flex-1 text-center py-1.5 bg-white rounded-md shadow-sm text-sm font-medium text-foreground">
                    English
                  </button>
                  <button className="flex-1 text-center py-1.5 text-muted-foreground text-sm hover:text-foreground transition-colors rounded-md">
                    Arabic
                  </button>
                </div>
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-[12px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Extraction Model
                </label>
                <select className="w-full bg-white border border-outline-variant rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary focus:ring-1 focus:ring-primary outline-none">
                  <option>Standard Financial (Fast)</option>
                  <option>Deep Analysis (Thorough)</option>
                  <option>Invoice Specific</option>
                </select>
              </div>
            </div>

            {/* Feature highlights card */}
            <div className="bg-secondary border border-accent rounded-xl p-4 flex flex-col gap-3 flex-grow">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Upload invoices, contracts, and bank statements. AI agents automatically
                detect contradictions and produce a bilingual audit report.
              </p>
              <ul className="space-y-2 text-xs text-muted-foreground">
                {[
                  "Cross-document contradiction detection",
                  "Arabic & English OCR support",
                  "Live chain-of-thought reasoning",
                  "Graph-based entity reconciliation",
                ].map((feat) => (
                  <li key={feat} className="flex items-start gap-2">
                    <span className="material-symbols-outlined text-primary text-[14px] mt-0.5">check_circle</span>
                    {feat}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </main>

      {/* Material Symbols font */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');`}</style>
    </div>
  );
}
