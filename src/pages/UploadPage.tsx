import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, FileImage, Loader2, AlertTriangle, CheckCircle,
  XCircle, Cpu, FileText, ChevronDown, ChevronUp, WifiOff, Info
} from "lucide-react";
import { type AnalysisResult } from "@/lib/mockApi";
import Navbar from "@/components/Navbar";
import AnimatedBackground from "@/components/AnimatedBackground";
import GlowCursor from "@/components/GlowCursor";
import Footer from "@/components/Footer";
import { Link } from "react-router-dom";
import { API_BASE_URL } from "../config";

interface ExtendedAnalysisResult extends AnalysisResult {
  ela_image?: string;
  elaApplicable?: boolean;
  metadata?: {
    software: string | null;
    has_exif: boolean;
    details: Record<string, string>;
  };
}

// Tooltip component for score explanations
const ScoreTooltip = ({ label, description }: { label: string; description: string }) => (
  <span className="group relative inline-flex items-center gap-1 cursor-help">
    <span>{label}</span>
    <Info className="h-3 w-3 text-muted-foreground" />
    <span className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-56 -translate-x-1/2 rounded bg-popover border border-border p-2 text-[10px] text-muted-foreground opacity-0 shadow-lg transition-opacity group-hover:opacity-100 leading-normal">
      {description}
    </span>
  </span>
);

const UploadPage = () => {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExtendedAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // FIX #1: backendOffline tracks whether the backend is unreachable.
  // When true, we show a clear error — we never silently return fake results.
  const [backendOffline, setBackendOffline] = useState(false);
  const [showMethodology, setShowMethodology] = useState(false);

  const handleFile = (f: File) => {
    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!validTypes.includes(f.type)) {
      setError("Unsupported file format. Please upload a JPG, PNG, or WEBP image.");
      setFile(null);
      setResult(null);
      return;
    }
    const maxSize = 10 * 1024 * 1024;
    if (f.size > maxSize) {
      setError("File is too large. Maximum supported size is 10 MB.");
      setFile(null);
      setResult(null);
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    setBackendOffline(false);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  }, []);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setBackendOffline(false);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { "X-Requested-With": "sheguard-client" },
        body: formData,
      });

      if (!response.ok) {
        // Show the server's error message if available
        let detail = response.statusText;
        try {
          const errBody = await response.json();
          if (errBody.detail) detail = errBody.detail;
        } catch { /* ignore */ }
        throw new Error(detail);
      }

      const res = await response.json();
      setResult(res);

      // Cache real results to localStorage (no fake data ever stored)
      const cached = localStorage.getItem("sheguard_cases");
      const casesList = cached ? JSON.parse(cached) : [];
      casesList.unshift({
        caseId: res.caseId,
        imageStatus: res.imageStatus,
        confidenceScore: res.confidenceScore,
        forensicScore: res.forensicScore,
        timestamp: res.timestamp || new Date().toISOString(),
        riskLevel: res.riskLevel,
        details: res.details,
        metadata: res.metadata,
        ela_image: res.ela_image,
        isDemo: false,  // Always false for real results
      });
      localStorage.setItem("sheguard_cases", JSON.stringify(casesList));

    } catch (err: any) {
      // FIX #1: When the backend is unreachable we show a clear error.
      // We do NOT silently fall back to random number generation.
      const msg: string = err?.message || "";
      const isNetworkError =
        msg.includes("Failed to fetch") ||
        msg.includes("NetworkError") ||
        msg.includes("ERR_CONNECTION") ||
        msg.includes("offline");

      if (isNetworkError) {
        setBackendOffline(true);
        setError(
          "The analysis server is currently offline. Real forensic analysis requires " +
          "the backend to be running. Please try again in a few moments."
        );
      } else {
        setError(`Analysis failed: ${msg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const riskIcon = (level: string) => {
    if (level === "green") return <CheckCircle className="h-5 w-5 risk-safe" />;
    if (level === "yellow") return <AlertTriangle className="h-5 w-5 risk-suspicious" />;
    return <XCircle className="h-5 w-5 risk-manipulated" />;
  };

  const riskClass = (level: string) =>
    level === "green" ? "progress-safe" : level === "yellow" ? "progress-suspicious" : "progress-manipulated";

  const riskTextClass = (level: string) =>
    level === "green" ? "risk-safe" : level === "yellow" ? "risk-suspicious" : "risk-manipulated";

  return (
    <div className="relative min-h-screen cyber-grid">
      <AnimatedBackground />
      <GlowCursor />
      <Navbar />
      <main className="mx-auto max-w-4xl px-4 pb-24 pt-28">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-2 text-center font-display text-3xl font-bold tracking-wide"
        >
          <span className="gradient-text">Image Analysis</span>
        </motion.h1>
        <p className="mb-10 text-center text-muted-foreground">
          Upload an image to detect manipulation
        </p>

        {/* Upload area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`glass-card cursor-pointer border-2 border-dashed p-12 text-center transition-all duration-300 relative overflow-hidden ${
            dragging
              ? "border-neon-purple bg-neon-purple/5 shadow-[0_0_15px_rgba(168,85,247,0.3)] scale-[1.01] animate-pulse"
              : "border-border hover:border-neon-blue/50 hover:bg-neon-blue/5"
          }`}
          onClick={() => !loading && document.getElementById("file-input")?.click()}
          role="button"
          aria-label="Click or drag and drop an image to upload for forensic analysis"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && !loading && document.getElementById("file-input")?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/webp"
            aria-label="Upload image file"
            className="hidden"
            disabled={loading}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />

          {loading ? (
            <div className="flex flex-col items-center gap-4 py-4 relative z-10">
              <div className="relative w-24 h-24 flex items-center justify-center">
                <FileImage className="h-16 w-16 text-neon-blue animate-pulse" />
                <div className="absolute inset-0 border-2 border-neon-blue border-t-neon-purple rounded-full animate-spin" />
              </div>
              <div className="space-y-1">
                <p className="font-display text-sm font-semibold tracking-wide text-neon-blue uppercase animate-pulse">
                  Running Forensic Analysis...
                </p>
                <p className="text-xs text-muted-foreground font-mono">
                  Analyzing pixels, noise patterns, and metadata
                </p>
              </div>
            </div>
          ) : file ? (
            <div className="flex flex-col items-center gap-3">
              <FileImage className="h-12 w-12 text-neon-purple" />
              <p className="font-body text-foreground">{file.name}</p>
              <p className="text-sm text-muted-foreground">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Upload className="h-12 w-12 text-muted-foreground" aria-hidden="true" />
              <p className="text-foreground">Drag & drop an image or click to browse</p>
              <p className="text-sm text-muted-foreground">Supports JPG, PNG, WEBP (Max 10 MB)</p>
            </div>
          )}
        </motion.div>

        <div className="mt-6 flex justify-center">
          <button
            onClick={analyze}
            disabled={!file || loading}
            aria-busy={loading}
            className="btn-glow flex items-center gap-2 font-display text-sm tracking-wide text-primary-foreground disabled:opacity-40"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : null}
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>
        </div>

        {/* Error / Offline state */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            role="alert"
            className="mt-6 glass-card border-destructive/50 p-4 text-center text-destructive"
          >
            {backendOffline
              ? <WifiOff className="mx-auto mb-2 h-6 w-6 text-destructive" aria-hidden="true" />
              : <AlertTriangle className="mx-auto mb-2 h-6 w-6 text-destructive" aria-hidden="true" />
            }
            <p className="font-semibold">
              {backendOffline ? "Analysis Server Offline" : "Analysis Error"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">{error}</p>
            {backendOffline && (
              <p className="mt-3 text-xs text-muted-foreground border-t border-border/30 pt-3">
                Unlike some demo tools, SheGuard AI does not generate fake scores when the server
                is unavailable. Only real forensic analysis results will be shown.
              </p>
            )}
          </motion.div>
        )}

        {/* Results */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-12"
          >
            <h2 className="mb-6 text-center font-display text-xl font-bold tracking-wide">
              Analysis <span className="gradient-text">Results</span>
            </h2>

            <div className="grid gap-6 md:grid-cols-2">
              {/* Left Column */}
              <div className="space-y-6">
                {/* Status card */}
                <div className="glass-card neon-border p-6">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Case ID</p>
                      <p className="font-display text-sm font-bold">{result.caseId}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      {riskIcon(result.riskLevel)}
                      <span className={`font-display text-lg font-bold ${riskTextClass(result.riskLevel)}`}>
                        {result.imageStatus}
                      </span>
                    </div>
                  </div>

                  <div className="grid gap-4 grid-cols-2 sm:grid-cols-3">
                    {[
                      {
                        label: "Confidence",
                        value: result.confidenceScore,
                        tip: "How certain the algorithm is about the verdict, based on the number and strength of forensic signals detected. This is NOT purely based on image resolution."
                      },
                      {
                        label: "Forensic Score",
                        value: result.forensicScore,
                        tip: "The highest anomaly score across all individual detectors (ELA, noise, face, metadata). Higher = more evidence of manipulation detected."
                      },
                    ].map((m) => (
                      <div key={m.label}>
                        <p className="mb-1 text-xs text-muted-foreground">
                          <ScoreTooltip label={m.label} description={m.tip} />
                        </p>
                        <div className="h-2 rounded-full bg-secondary" role="progressbar" aria-valuenow={m.value} aria-valuemin={0} aria-valuemax={100}>
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${m.value}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className={`h-full rounded-full ${riskClass(result.riskLevel)}`}
                          />
                        </div>
                        <p className="mt-1 text-right text-xs text-muted-foreground">{m.value}%</p>
                      </div>
                    ))}
                    <div className="col-span-2 sm:col-span-1">
                      <p className="mb-1 text-xs text-muted-foreground">Timestamp</p>
                      <p className="text-xs text-foreground mt-1">
                        {new Date(result.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Detailed scores */}
                <div className="glass-card p-6">
                  <h3 className="mb-1 font-display text-sm font-semibold tracking-wide">
                    Forensic Detail Scores
                  </h3>
                  <p className="mb-4 text-[10px] text-muted-foreground leading-normal">
                    Each score measures a specific forensic indicator (0–100). Higher scores indicate
                    a greater statistical anomaly in that dimension, not a definitive finding.
                  </p>
                  <div className="space-y-3">
                    {Object.entries(result.details).map(([key, val]) => (
                      <div key={key}>
                        <div className="mb-1 flex justify-between text-xs">
                          <span className="text-muted-foreground capitalize">
                            {key.replace(/([A-Z])/g, " $1")}
                          </span>
                          <span className="text-foreground">{val}%</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-secondary" role="progressbar" aria-valuenow={val as number} aria-valuemin={0} aria-valuemax={100}>
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${val}%` }}
                            transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
                            className="h-full rounded-full"
                            style={{ background: "linear-gradient(90deg, hsl(270 80% 65%), hsl(200 100% 55%))" }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Column */}
              <div className="space-y-6">
                {/* ELA Visualizer */}
                <div className="glass-card p-6">
                  <div className="flex items-center gap-2 mb-3">
                    <Cpu className="h-4 w-4 text-neon-blue" aria-hidden="true" />
                    <h3 className="font-display text-sm font-semibold tracking-wide">
                      Error Level Analysis (ELA) Scan
                    </h3>
                  </div>
                  <div className="relative overflow-hidden rounded border border-border bg-black/80 p-2 flex justify-center items-center h-[236px]">
                    {result.ela_image ? (
                      <img
                        src={`data:image/jpeg;base64,${result.ela_image}`}
                        alt="Error Level Analysis visualization showing compression anomalies"
                        className="max-h-[220px] object-contain rounded"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center border border-dashed border-neon-blue/30 rounded p-4 text-center">
                        <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
                          {result.elaApplicable === false
                            ? "ELA Not Applicable"
                            : "ELA Unavailable"}
                        </p>
                        {/* FIX #16: No fake spinner — honest message about why ELA isn't shown */}
                        <p className="text-[10px] text-muted-foreground mt-2 leading-normal">
                          {result.elaApplicable === false
                            ? "ELA is a JPEG-specific technique. This PNG or WEBP image was analyzed using noise and metadata signals instead."
                            : "ELA visualization could not be generated for this image."}
                        </p>
                      </div>
                    )}
                  </div>
                  {/* FIX #11: Explain when ELA doesn't apply */}
                  <p className="mt-2 text-xs text-muted-foreground leading-normal">
                    {result.elaApplicable !== false
                      ? "High-contrast areas indicate compression rate variance, a common sign of splicing or editing."
                      : "Note: ELA analysis is only valid for JPEG images. PNG and WEBP files use lossless compression where ELA would produce meaningless results."}
                  </p>
                </div>

                {/* EXIF Metadata */}
                <div className="glass-card p-6">
                  <div className="flex items-center gap-2 mb-3">
                    <FileText className="h-4 w-4 text-neon-purple" aria-hidden="true" />
                    <h3 className="font-display text-sm font-semibold tracking-wide">
                      EXIF Metadata Analysis
                    </h3>
                  </div>

                  {result.metadata ? (
                    <div className="space-y-3">
                      <div className="flex justify-between items-center rounded bg-secondary/30 p-2.5 text-xs">
                        <span className="text-muted-foreground">Software Trace</span>
                        {result.metadata.software ? (
                          <span className="font-bold text-destructive animate-pulse bg-destructive/10 px-2 py-0.5 rounded border border-destructive/20">
                            {result.metadata.software} detected
                          </span>
                        ) : (
                          <span className="text-emerald-400 font-semibold">No editor signature</span>
                        )}
                      </div>

                      <div className="max-h-[140px] overflow-y-auto text-xs space-y-1 pr-1">
                        {Object.keys(result.metadata.details).length > 0 ? (
                          Object.entries(result.metadata.details).map(([k, v]) => (
                            <div key={k} className="flex justify-between py-1 border-b border-border/20">
                              <span className="text-muted-foreground">{k}</span>
                              <span className="text-foreground font-mono truncate max-w-[150px]">{v}</span>
                            </div>
                          ))
                        ) : (
                          <p className="text-muted-foreground text-center py-4">
                            No EXIF tags found — metadata may have been stripped.
                          </p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">EXIF audit unavailable.</p>
                  )}
                </div>
              </div>
            </div>

            {/* View Evidence Report button */}
            <div className="mt-8 flex justify-center">
              <Link
                to={`/evidence-report?caseId=${result.caseId}&status=${result.imageStatus}&confidence=${result.confidenceScore}&forensic=${result.forensicScore}&risk=${result.riskLevel}`}
                className="btn-glow font-display text-sm tracking-wide text-primary-foreground"
              >
                View Evidence Report
              </Link>
            </div>
          </motion.div>
        )}

        {/* Algorithm Methodology */}
        <div className="mt-12 border-t border-border/40 pt-8">
          <button
            onClick={() => setShowMethodology(!showMethodology)}
            aria-expanded={showMethodology}
            className="mx-auto flex items-center gap-2 text-xs uppercase tracking-widest text-muted-foreground hover:text-foreground transition-colors"
          >
            <span>Forensics Methodology & Limitations</span>
            {showMethodology
              ? <ChevronUp className="h-4 w-4" aria-hidden="true" />
              : <ChevronDown className="h-4 w-4" aria-hidden="true" />}
          </button>

          <AnimatePresence>
            {showMethodology && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden mt-6 text-xs text-muted-foreground leading-relaxed space-y-4 max-w-3xl mx-auto bg-secondary/10 p-5 rounded-lg border border-border/55"
              >
                <h4 className="font-display font-semibold text-foreground uppercase tracking-wide">
                  Algorithm Transparency Disclosure
                </h4>
                <p>
                  SheGuard AI uses three classical forensic techniques to identify statistical
                  anomalies. Each produces a 0–100 score reflecting the strength of the anomaly
                  detected — not a probability of manipulation.
                </p>
                <div className="grid gap-4 sm:grid-cols-3 mt-2">
                  <div className="space-y-1">
                    <h5 className="font-semibold text-neon-purple uppercase">1. Error Level Analysis (ELA)</h5>
                    <p className="text-[11px]">
                      Re-saves the image at 95% JPEG quality and measures pixel-level differences.
                      Regions that were edited compress at different rates, producing visible edges.
                      <strong className="text-yellow-400"> Only valid for JPEG images.</strong>
                    </p>
                  </div>
                  <div className="space-y-1">
                    <h5 className="font-semibold text-neon-blue uppercase">2. Local Noise Check</h5>
                    <p className="text-[11px]">
                      Extracts high-frequency noise from flat image regions using a median filter.
                      Inconsistent noise variance between regions can indicate compositing from
                      different sources.
                    </p>
                  </div>
                  <div className="space-y-1">
                    <h5 className="font-semibold text-neon-pink uppercase">3. Face Swap Detection</h5>
                    <p className="text-[11px]">
                      Uses Haar Cascade face detection (classic CV, not deep learning) and compares
                      the ELA compression signature inside detected faces against the background.
                      <strong className="text-yellow-400"> Limited to frontal faces and may miss angled or partial faces.</strong>
                    </p>
                  </div>
                </div>
                <div className="border border-yellow-500/30 bg-yellow-500/10 rounded p-3 mt-2">
                  <p className="text-[11px] text-yellow-400 font-semibold">
                    ⚠ Important Limitations
                  </p>
                  <p className="text-[11px] mt-1">
                    These are signal-based heuristics, not a certified forensic standard. A "Safe"
                    result does not guarantee authenticity — sophisticated manipulations may evade
                    all three detectors. A "Manipulated" result does not prove malicious intent —
                    legitimate photo editing (cropping, brightness, filters) can also trigger these
                    signals. Results must be verified by a qualified forensic examiner before being
                    used as evidence.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default UploadPage;
