"use client";

import { useState, useCallback } from "react";
import { useDropzone, type DropEvent, type FileRejection } from "react-dropzone";
import { Upload, FileText, X, AlertCircle, Languages, Receipt, FileBadge2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { uploadDocuments } from "@/lib/api";
import type { DocumentMeta, UploadResponse } from "@/types";

interface FileUploadProps {
  onUploadComplete: (response: UploadResponse) => void;
}

const MAX_SIZE_MB = 50;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export function FileUpload({ onUploadComplete }: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [reportLanguage, setReportLanguage] = useState<"english" | "arabic">("english");
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    <T extends File>(accepted: T[], rejected: FileRejection[], _event: DropEvent) => {
      setError(null);
      if (rejected.length > 0) {
        setError(
          `Some files were rejected: ${rejected.map((r) => r.file.name).join(", ")}. Only PDFs under ${MAX_SIZE_MB}MB are accepted.`
        );
      }
      setFiles((prev) => {
        const existing = new Set(prev.map((f) => f.name));
        const newFiles = accepted.filter((f) => !existing.has(f.name));
        return [...prev, ...newFiles];
      });
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxSize: MAX_SIZE_BYTES,
    multiple: true,
  });

  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setIsUploading(true);
    setError(null);
    try {
      const response = await uploadDocuments(files, reportLanguage);
      onUploadComplete(response);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 relative overflow-hidden",
          isDragActive
            ? "border-primary bg-primary/10 scale-[1.01] shadow-[0_0_0_3px_rgba(59,130,246,0.15)]"
            : "border-border hover:border-primary/50 hover:bg-accent/30"
        )}
      >
        <input {...getInputProps()} />
        {isDragActive && (
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />
        )}
        <div className="flex flex-col items-center gap-3">
          <div className="p-3 rounded-full bg-primary/10">
            <Upload className="h-6 w-6 text-primary" />
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">
              {isDragActive ? "Drop your PDFs here..." : "Drag & drop PDFs here"}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              or click to browse — Arabic & English documents supported
            </p>
          </div>
          <p className="text-xs text-muted-foreground">
            PDF only · Max {MAX_SIZE_MB}MB per file · Multiple files allowed
          </p>
          <div className="flex items-center gap-2 pt-1">
            <span className="inline-flex items-center gap-1 text-[10px] px-2 py-1 rounded-full border border-border bg-secondary/60 text-muted-foreground">
              <FileBadge2 className="h-3 w-3 text-red-400" />
              PDF
            </span>
            <span className="inline-flex items-center gap-1 text-[10px] px-2 py-1 rounded-full border border-border bg-secondary/60 text-muted-foreground">
              <Receipt className="h-3 w-3 text-primary" />
              Invoice / Contract
            </span>
          </div>
        </div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file) => (
            <div
              key={file.name}
              className="flex items-center gap-3 px-3 py-2 rounded-lg bg-secondary/50 border border-border"
            >
              <FileText className="h-4 w-4 text-primary shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={() => removeFile(file.name)}
                className="p-1 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive transition-colors"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Report language selector */}
      <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30 border border-border">
        <Languages className="h-4 w-4 text-primary shrink-0" />
        <span className="text-sm text-muted-foreground">Report language:</span>
        <div className="flex gap-2 ml-auto">
          {(["english", "arabic"] as const).map((lang) => (
            <button
              key={lang}
              onClick={() => setReportLanguage(lang)}
              className={cn(
                "px-3 py-1 rounded text-xs font-medium transition-all",
                reportLanguage === lang
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-muted-foreground hover:bg-accent"
              )}
            >
              {lang === "english" ? "🇬🇧 English" : "🇪🇬 Arabic"}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm">
          <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={files.length === 0 || isUploading}
        className={cn(
          "w-full py-3 px-4 rounded-xl font-medium text-sm transition-all duration-200",
          files.length === 0 || isUploading
            ? "bg-secondary text-muted-foreground cursor-not-allowed"
            : "bg-gradient-to-r from-primary to-blue-500 text-white hover:brightness-110 shadow-lg shadow-primary/30 hover:shadow-primary/50"
        )}
      >
        {isUploading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="h-4 w-4 rounded-full border-2 border-primary-foreground border-t-transparent animate-spin" />
            Processing documents...
          </span>
        ) : (
          `Start Audit${files.length > 0 ? ` (${files.length} file${files.length > 1 ? "s" : ""})` : ""}`
        )}
      </button>
    </div>
  );
}
