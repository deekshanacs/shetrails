import { motion } from "framer-motion";
import { useSearchParams } from "react-router-dom";
import { Download, Shield, FileText, Scale, Loader2, Info } from "lucide-react";
import Navbar from "@/components/Navbar";
import AnimatedBackground from "@/components/AnimatedBackground";
import GlowCursor from "@/components/GlowCursor";
import Footer from "@/components/Footer";
import { useEffect, useState } from "react";
import { API_BASE_URL } from "../config";

// FIX #2 & #3: Legal recommendation is now conditional on the actual verdict.
// It no longer always says "signs of digital manipulation."
function getLegalRecommendation(status: string): {
  text: string;
  links: { label: string; url: string }[];
  severity: "safe" | "suspicious" | "manipulated";
} {
  const s = status.toLowerCase();

  if (s === "manipulated") {
    return {
      severity: "manipulated",
      text:
        "The forensic analysis detected significant statistical anomalies consistent with " +
        "digital manipulation. If this image was used in a harmful context (e.g. non-consensual " +
        "intimate imagery, defamation, or fraud), you may consider filing a formal complaint. " +
        "Always verify AI findings with a certified forensic examiner before legal action.",
      links: [
        { label: "India: National Cybercrime Portal", url: "https://cybercrime.gov.in" },
        { label: "UK: Report to Action Fraud", url: "https://www.actionfraud.police.uk" },
        { label: "US: FBI Internet Crime Complaint Center", url: "https://www.ic3.gov" },
        { label: "EU: Europol Report", url: "https://www.europol.europa.eu/report-a-crime/report-cybercrime-online" },
      ],
    };
  }

  if (s === "suspicious") {
    return {
      severity: "suspicious",
      text:
        "The analysis found moderate anomalies. This may result from legitimate editing " +
        "(e.g. filters, cropping, brightness adjustments) or from actual manipulation. " +
        "The result is inconclusive. Do not use this report as evidence without further " +
        "examination by a certified forensic professional.",
      links: [
        { label: "Find a certified forensic examiner", url: "https://www.acfe.com/fraud-examiners.aspx" },
      ],
    };
  }

  // Safe
  return {
    severity: "safe",
    text:
      "No significant forensic anomalies were detected. The image appears consistent with " +
      "an unedited photograph based on the signals analyzed. Note: a \"Safe\" result does not " +
      "guarantee authenticity — sophisticated editing tools may evade these detectors. " +
      "This report should not be used as legal proof of authenticity.",
    links: [],
  };
}

const EvidenceReport = () => {
  const [params] = useSearchParams();
  const caseId = params.get("caseId") || "SG-DEMO";

  const [status, setStatus]       = useState(params.get("status") || "");
  const [confidence, setConfidence] = useState(params.get("confidence") || "");
  const [forensic, setForensic]   = useState(params.get("forensic") || "");
  const [risk, setRisk]           = useState(params.get("risk") || "");
  const [loading, setLoading]     = useState(false);

  useEffect(() => {
    if (params.get("status") && params.get("confidence") && params.get("forensic") && params.get("risk")) {
      return;
    }

    const loadCase = async () => {
      setLoading(true);

      // Try local storage cache first
      const cached = localStorage.getItem("sheguard_cases");
      if (cached) {
        const localCases = JSON.parse(cached);
        const match = localCases.find((c: any) => c.caseId === caseId);
        if (match) {
          setStatus(match.imageStatus);
          setConfidence(String(match.confidenceScore));
          setForensic(String(match.forensicScore));
          setRisk(match.riskLevel);
          setLoading(false);
          return;
        }
      }

      // Try backend API
      try {
        const response = await fetch(`${API_BASE_URL}/api/cases`);
        if (response.ok) {
          const data = await response.json();
          const match = data.find((c: any) => c.caseId === caseId);
          if (match) {
            setStatus(match.imageStatus);
            setConfidence(String(match.confidenceScore));
            setForensic(String(match.forensicScore));
            setRisk(match.riskLevel);
          }
        }
      } catch (err) {
        console.error("Failed to fetch case details", err);
      } finally {
        setLoading(false);
      }
    };

    loadCase();
  }, [caseId, params]);

  const displayStatus     = status || "Unknown";
  const displayConfidence = confidence || "—";
  const displayForensic   = forensic || "—";
  const displayRisk       = risk || "green";

  const riskTextClass =
    displayRisk === "green" ? "risk-safe" :
    displayRisk === "yellow" ? "risk-suspicious" :
    "risk-manipulated";

  // FIX #2: Legal recommendation is derived from the REAL status
  const recommendation = getLegalRecommendation(displayStatus);

  // FIX #3: Download report includes all real data and correct recommendation
  const handleDownload = () => {
    const legalSection =
      recommendation.severity === "manipulated"
        ? `LEGAL RECOMMENDATION
This image shows forensic indicators of digital manipulation. If used in a harmful
context, consider filing a complaint with the appropriate authority listed below.

Reporting portals:
${recommendation.links.map(l => `  - ${l.label}: ${l.url}`).join("\n")}

DISCLAIMER: AI analysis results are indicative only. Verification by a certified
forensic examiner is required before using this report as legal evidence.`
        : recommendation.severity === "suspicious"
        ? `LEGAL RECOMMENDATION
The result is inconclusive. Moderate anomalies were detected but may be due to
legitimate photo editing. Do not rely on this report without professional examination.`
        : `LEGAL RECOMMENDATION
No significant manipulation detected. A "Safe" result does not guarantee authenticity.
Do not use this report as legal proof of authenticity.`;

    const content = `
SHE-GUARD AI — CYBER FORENSIC ANALYSIS REPORT
===============================================

CASE INFORMATION
  Case ID          : ${caseId}
  Date             : ${new Date().toLocaleDateString()}
  Time             : ${new Date().toLocaleTimeString()}
  Analysis System  : SheGuard AI v1.0 (Classical Forensics Pipeline)

AI ANALYSIS RESULT
  Image Status     : ${displayStatus}
  Confidence Score : ${displayConfidence}%
  Forensic Score   : ${displayForensic}%

${legalSection}

===============================================
Generated by SheGuard AI | ${new Date().toISOString()}
`.trim();

    const blob = new Blob([content], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `SheGuard-Report-${caseId}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative min-h-screen cyber-grid">
      <AnimatedBackground />
      <GlowCursor />
      <Navbar />
      <main className="mx-auto max-w-3xl px-4 pb-24 pt-28">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="mb-8 flex items-center justify-center gap-3">
            <Shield className="h-8 w-8 text-neon-purple" aria-hidden="true" />
            <h1 className="font-display text-2xl font-bold tracking-wide sm:text-3xl">
              <span className="gradient-text">Evidence Report</span>
            </h1>
          </div>

          <div className="glass-card neon-border overflow-hidden">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 text-muted-foreground gap-3">
                <Loader2 className="h-10 w-10 animate-spin text-neon-purple" aria-hidden="true" />
                <p className="font-mono text-xs uppercase tracking-wider">
                  Locating Forensic Log {caseId}...
                </p>
              </div>
            ) : (
              <>
                {/* Header */}
                <div
                  className="border-b border-border p-6"
                  style={{ background: "linear-gradient(135deg, hsl(270 80% 65% / 0.1), hsl(200 100% 55% / 0.05))" }}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-wider text-muted-foreground">Case ID</p>
                      <p className="font-display text-lg font-bold">{caseId}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs uppercase tracking-wider text-muted-foreground">Date</p>
                      <p className="text-sm">{new Date().toLocaleDateString()}</p>
                    </div>
                  </div>
                </div>

                <div className="divide-y divide-border">
                  <Section icon={FileText} title="Case Information">
                    <InfoRow label="Case ID"          value={caseId} />
                    <InfoRow label="Report Generated" value={new Date().toLocaleString()} />
                    <InfoRow label="System"           value="SheGuard AI v1.0" />
                  </Section>

                  <Section icon={Shield} title="AI Analysis Result">
                    <InfoRow label="Image Status"     value={displayStatus} valueClass={riskTextClass} />
                    <InfoRow label="Confidence Score" value={displayConfidence !== "—" ? `${displayConfidence}%` : "—"} />
                    <InfoRow label="Forensic Score"   value={displayForensic  !== "—" ? `${displayForensic}%`  : "—"} />
                  </Section>

                  {/* FIX #2: Conditional legal recommendation based on actual verdict */}
                  <Section icon={Scale} title="Legal Recommendation">
                    {/* Status-specific colour border */}
                    <div className={`rounded p-3 text-xs leading-relaxed mb-3 border ${
                      recommendation.severity === "manipulated"
                        ? "border-destructive/30 bg-destructive/10 text-destructive"
                        : recommendation.severity === "suspicious"
                        ? "border-yellow-500/30 bg-yellow-500/10 text-yellow-400"
                        : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                    }`}>
                      <p className="font-semibold mb-1 flex items-center gap-1">
                        <Info className="h-3 w-3" aria-hidden="true" />
                        {recommendation.severity === "manipulated" && "Manipulation Indicators Found"}
                        {recommendation.severity === "suspicious"  && "Inconclusive — Further Examination Needed"}
                        {recommendation.severity === "safe"        && "No Significant Anomalies Detected"}
                      </p>
                      <p>{recommendation.text}</p>
                    </div>

                    {recommendation.links.length > 0 && (
                      <div className="space-y-1 text-xs">
                        <p className="text-muted-foreground font-semibold">Reporting Resources:</p>
                        {recommendation.links.map((link) => (
                          <a
                            key={link.url}
                            href={link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block text-neon-blue underline hover:text-neon-blue/80"
                          >
                            {link.label}
                          </a>
                        ))}
                      </div>
                    )}

                    <p className="mt-3 text-[10px] text-muted-foreground italic">
                      This AI report is for informational use only. It is not a certified forensic
                      document and cannot be used as standalone evidence in legal proceedings.
                    </p>
                  </Section>
                </div>

                {/* Download */}
                <div className="p-6">
                  <button
                    onClick={handleDownload}
                    className="btn-glow flex w-full items-center justify-center gap-2 font-display text-sm tracking-wide text-primary-foreground"
                  >
                    <Download className="h-4 w-4" aria-hidden="true" />
                    Download Evidence Report
                  </button>
                </div>
              </>
            )}
          </div>
        </motion.div>
      </main>
      <Footer />
    </div>
  );
};

const Section = ({
  icon: Icon,
  title,
  children,
}: {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
}) => (
  <div className="p-6">
    <div className="mb-3 flex items-center gap-2">
      <Icon className="h-4 w-4 text-neon-purple" aria-hidden="true" />
      <h3 className="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
    </div>
    <div className="space-y-2">{children}</div>
  </div>
);

const InfoRow = ({
  label,
  value,
  valueClass,
}: {
  label: string;
  value: string;
  valueClass?: string;
}) => (
  <div className="flex justify-between text-sm">
    <span className="text-muted-foreground">{label}</span>
    <span className={valueClass || "text-foreground font-medium"}>{value}</span>
  </div>
);

export default EvidenceReport;
