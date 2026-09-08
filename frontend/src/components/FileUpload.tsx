"use client";

import { useState, useCallback } from "react";
import { useDropzone, type DropEvent, type FileRejection } from "react-dropzone";
import { AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { uploadDocuments } from "@/lib/api";
import type { UploadResponse } from "@/types";

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
  const [partialUpload, setPartialUpload] = useState<UploadResponse | null>(null);

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
    setPartialUpload(null);
    try {
      const response = await uploadDocuments(files, reportLanguage);
      if (response.failed.length > 0) {
        setPartialUpload(response);
      } else {
        onUploadComplete(response);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-4 w-full">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-all",
          isDragActive
            ? "border-primary bg-secondary"
            : "border-outline-variant hover:border-primary/50 hover:bg-secondary/60"
        )}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-3">
          <div className="w-14 h-14 bg-muted rounded-full flex items-center justify-center">
            <span className="material-symbols-outlined text-[28px] text-primary">upload_file</span>
          </div>
          <div>
            <p className="text-base font-semibold text-foreground">
              {isDragActive ? "Drop your files here..." : "Drag and drop your files here"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Support for PDF up to {MAX_SIZE_MB}MB per file.
            </p>
          </div>
          {/* No onClick — the click bubbles to the dropzone root, which opens the picker. */}
          <button
            type="button"
            className="bg-[#131b2e] text-white text-sm font-medium px-5 py-2 rounded-lg hover:opacity-90 transition-opacity"
          >
            Browse Files
          </button>
        </div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-foreground">Uploaded Documents</h3>
          {files.map((file) => (
            <div
              key={file.name}
              className="flex items-center justify-between p-3 border border-outline-variant rounded-lg bg-white hover:bg-secondary transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-[20px] text-muted-foreground">description</span>
                <div>
                  <p className="text-sm font-semibold text-foreground truncate max-w-[200px]">{file.name}</p>
                  <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="inline-flex items-center gap-1 bg-green-50 text-green-700 border border-green-200 px-2 py-0.5 rounded-full text-[10px] font-semibold">
                  <span className="material-symbols-outlined text-[12px]">check_circle</span>
                  Ready
                </span>
                <button
                  onClick={() => removeFile(file.name)}
                  className="text-muted-foreground hover:text-destructive transition-colors"
                >
                  <span className="material-symbols-outlined text-[20px]">delete</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Report language */}
      <div className="flex items-center gap-3">
        <span className="text-[12px] font-semibold uppercase tracking-wide text-muted-foreground">
          Report Language:
        </span>
        <div className="flex gap-2">
          {(["english", "arabic"] as const).map((lang) => (
            <button
              key={lang}
              onClick={() => setReportLanguage(lang)}
              className={cn(
                "px-3 py-1.5 rounded-lg text-sm font-medium transition-all border",
                reportLanguage === lang
                  ? "bg-primary text-white border-primary"
                  : "bg-white text-muted-foreground border-outline-variant hover:bg-secondary"
              )}
            >
              {lang === "english" ? "English" : "Arabic"}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
          <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Partial upload failure — some files processed, some didn't */}
      {partialUpload && (
        <div className="space-y-2 p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-sm">
          <div className="flex items-start gap-2">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <p>
              {partialUpload.documents.length} of {files.length} file(s) processed successfully.{" "}
              {partialUpload.failed.length} failed and will be excluded from the audit:
            </p>
          </div>
          <ul className="pl-6 list-disc space-y-0.5">
            {partialUpload.failed.map((f) => (
              <li key={f.filename}>
                <span className="font-semibold">{f.filename}</span>: {f.error}
              </li>
            ))}
          </ul>
          <button
            onClick={() => onUploadComplete(partialUpload)}
            className="mt-1 bg-amber-800 text-white text-xs font-medium px-3 py-1.5 rounded-md hover:opacity-90 transition-opacity"
          >
            Continue with {partialUpload.documents.length} document{partialUpload.documents.length > 1 ? "s" : ""}
          </button>
        </div>
      )}

      {/* Start Audit button */}
      <button
        onClick={handleUpload}
        disabled={files.length === 0 || isUploading}
        className={cn(
          "w-full py-3 px-4 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-2",
          files.length === 0 || isUploading
            ? "bg-muted text-muted-foreground cursor-not-allowed"
            : "bg-primary text-white hover:bg-[#003ea8]"
        )}
      >
        {isUploading ? (
          <>
            <span className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
            Processing documents...
          </>
        ) : (
          <>
            <span className="material-symbols-outlined text-[18px]">play_arrow</span>
            {`Start Audit${files.length > 0 ? ` (${files.length} file${files.length > 1 ? "s" : ""})` : ""}`}
          </>
        )}
      </button>
    </div>
  );
}

