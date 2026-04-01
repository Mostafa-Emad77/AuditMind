"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { X, FileText } from "lucide-react";
import type { Finding } from "@/types";
import { cn, severityBg } from "@/lib/utils";

/** Parse "filename, page N: snippet" style evidence lines. */
export function parseEvidenceLine(line: string): {
  filename: string;
  page?: number;
  quote: string;
} {
  const m = line.match(/^(.+?),\s*page\s+(\d+)\s*:\s*(.*)$/i);
  if (m) {
    return { filename: m[1].trim(), page: parseInt(m[2], 10), quote: m[3].trim() };
  }
  return { filename: "", quote: line.trim() };
}

function resolveFilename(
  raw: string,
  docFilenameById: Record<string, string>
): string {
  const s = raw.trim();
  for (const [id, name] of Object.entries(docFilenameById)) {
    if (s.includes(id.slice(0, 8))) return name;
  }
  return raw;
}

export function FindingDetailSheet({
  finding,
  open,
  onOpenChange,
  docFilenameById,
}: {
  finding: Finding | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  docFilenameById: Record<string, string>;
}) {
  if (!finding) return null;

  const sourceDocName = finding.source_doc_id
    ? docFilenameById[finding.source_doc_id] ||
      `${finding.source_doc_id.slice(0, 8)}…`
    : null;
  const conflictDocName = finding.conflicting_doc_id
    ? docFilenameById[finding.conflicting_doc_id] ||
      `${finding.conflicting_doc_id.slice(0, 8)}…`
    : null;

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-50" />
        <Dialog.Content className="fixed right-0 top-0 z-50 h-full w-full max-w-lg border-l border-border bg-background shadow-xl flex flex-col outline-none">
          <div className="flex items-start justify-between gap-3 p-4 border-b border-border">
            <div className="min-w-0">
              <Dialog.Title className="text-sm font-semibold leading-tight pr-6">
                {finding.title}
              </Dialog.Title>
              <span
                className={cn(
                  "inline-block mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full border",
                  severityBg(finding.severity)
                )}
              >
                {finding.severity.toUpperCase()}
              </span>
            </div>
            <Dialog.Close className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground shrink-0">
              <X className="h-4 w-4" />
            </Dialog.Close>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-5">
            <div>
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium mb-1">
                Description
              </p>
              <p
                className="text-sm text-foreground/90 leading-relaxed"
                dir={finding.description.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
              >
                {finding.description}
              </p>
            </div>

            {finding.evidence.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium mb-2">
                  Source passages
                </p>
                <ul className="space-y-3">
                  {finding.evidence.map((line, idx) => {
                    const parsed = parseEvidenceLine(line);
                    const label = `Source ${String.fromCharCode(65 + idx)}`;
                    const fname = parsed.filename
                      ? resolveFilename(parsed.filename, docFilenameById)
                      : "Document";
                    return (
                      <li
                        key={idx}
                        className="rounded-lg border border-border bg-secondary/30 p-3 text-sm"
                      >
                        <p className="text-xs font-semibold text-primary mb-1 flex items-center gap-1.5">
                          <FileText className="h-3.5 w-3.5 shrink-0" />
                          {label}
                          {parsed.filename && (
                            <span className="font-normal text-muted-foreground">
                              — {fname}
                              {parsed.page != null ? `, page ${parsed.page}` : ""}
                            </span>
                          )}
                        </p>
                        <p
                          className="text-xs text-foreground/80 leading-relaxed pl-5 border-l-2 border-primary/30"
                          dir={parsed.quote.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                        >
                          {parsed.quote || line}
                        </p>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {(sourceDocName || conflictDocName) && (
              <div className="text-xs space-y-1">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">
                  Document references
                </p>
                {sourceDocName && (
                  <p className="text-muted-foreground">
                    Primary: <span className="text-foreground font-mono">{sourceDocName}</span>
                    {finding.source_page != null ? ` · p.${finding.source_page}` : ""}
                  </p>
                )}
                {conflictDocName && (
                  <p className="text-muted-foreground">
                    Cross-doc: <span className="text-foreground font-mono">{conflictDocName}</span>
                    {finding.conflicting_page != null ? ` · p.${finding.conflicting_page}` : ""}
                  </p>
                )}
              </div>
            )}

            {finding.recommendation && (
              <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
                <p className="text-[10px] uppercase tracking-wide text-primary font-medium mb-1">
                  Recommendation
                </p>
                <p
                  className="text-xs text-primary/90"
                  dir={finding.recommendation.match(/[\u0600-\u06FF]/) ? "rtl" : "ltr"}
                >
                  {finding.recommendation}
                </p>
              </div>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
