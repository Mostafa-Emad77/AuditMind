"use client";

import Image from "next/image";
import Link from "next/link";
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
        <div className="flex items-center justify-between w-full max-w-[1440px] mx-auto">
          <div className="flex items-center gap-2 text-xl font-bold tracking-tight text-foreground">
            <Image src="/logo.png" alt="AuditMind Logo" width={32} height={32} className="object-contain" />
            AuditMind
          </div>
          <Link
            href="/archive"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            Archive
          </Link>
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
