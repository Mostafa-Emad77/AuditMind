import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AuditMind — Bilingual Financial Document Auditor",
  description:
    "AI-powered agentic auditor for Arabic and English financial documents. Cross-document contradiction detection with visible chain-of-thought reasoning.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background antialiased">
        {children}
      </body>
    </html>
  );
}
