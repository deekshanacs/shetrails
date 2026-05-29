import { useState } from "react";
import { motion } from "framer-motion";
import { Send, CheckCircle, Loader2, WifiOff, AlertTriangle } from "lucide-react";
import { type IncidentReport } from "@/lib/mockApi";
import Navbar from "@/components/Navbar";
import AnimatedBackground from "@/components/AnimatedBackground";
import GlowCursor from "@/components/GlowCursor";
import Footer from "@/components/Footer";
import { Link } from "react-router-dom";
import { API_BASE_URL } from "../config";

const FIELD_LIMITS: Record<string, number> = {
  name: 100,
  email: 254,
  location: 200,
  contact: 20,
};

const fields = [
  { name: "name",     label: "Full Name",       type: "text" },
  { name: "email",    label: "Email Address",   type: "email" },
  { name: "gender",   label: "Gender",          type: "select",
    options: ["Female", "Male", "Non-binary", "Prefer not to say"] },
  { name: "age",      label: "Age",             type: "number" },
  { name: "location", label: "Location",        type: "text" },
  { name: "contact",  label: "Contact Number",  type: "tel" },
] as const;

const ReportPage = () => {
  const [form, setForm] = useState({
    name: "", email: "", gender: "", age: "",
    location: "", contact: "", description: "",
  });
  const [consent, setConsent]   = useState(false);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState<IncidentReport | null>(null);
  const [error, setError]       = useState<string | null>(null);
  const [backendOffline, setBackendOffline] = useState(false);

  const update = (key: string, val: string) => {
    const limit = FIELD_LIMITS[key];
    setForm((f) => ({ ...f, [key]: limit ? val.slice(0, limit) : val }));
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!consent) {
      setError("You must consent to the privacy policy before submitting.");
      return;
    }
    setLoading(true);
    setError(null);
    setBackendOffline(false);

    try {
      const response = await fetch(`${API_BASE_URL}/api/reports`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Requested-With": "sheguard-client",
        },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        let detail = "Failed to submit report.";
        try {
          const errBody = await response.json();
          if (errBody.detail) detail = errBody.detail;
        } catch { /* ignore */ }
        throw new Error(detail);
      }

      const res = await response.json();
      setResult(res);
    } catch (err: any) {
      const msg: string = err?.message || "";
      const isNetworkError =
        msg.includes("Failed to fetch") ||
        msg.includes("NetworkError") ||
        msg.includes("ERR_CONNECTION");

      if (isNetworkError) {
        // FIX: Do NOT silently "simulate" a successful filing when offline.
        // The user's report was NOT received. Show a clear, honest error.
        setBackendOffline(true);
        setError(
          "Your report could not be submitted because the server is currently offline. " +
          "Your data has NOT been sent or saved anywhere. " +
          "Please try again later, or contact the helpline directly."
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen cyber-grid">
      <AnimatedBackground />
      <GlowCursor />
      <Navbar />
      <main className="mx-auto max-w-2xl px-4 pb-24 pt-28">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-2 text-center font-display text-3xl font-bold tracking-wide"
        >
          <span className="gradient-text-accent">Report Incident</span>
        </motion.h1>
        <p className="mb-10 text-center text-muted-foreground">
          Your identity will be protected
        </p>

        {result ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card neon-border p-8 text-center"
            role="status"
          >
            <CheckCircle className="mx-auto mb-4 h-12 w-12 risk-safe" aria-hidden="true" />
            <h2 className="mb-2 font-display text-xl font-bold">Report Filed Successfully</h2>
            <p className="mb-4 text-muted-foreground">Your case ID is:</p>
            <p className="font-display text-2xl font-bold gradient-text">{result.caseId}</p>
            <p className="mt-4 text-xs text-muted-foreground">
              Status: {result.status} &bull; {new Date(result.submittedAt).toLocaleString()}
            </p>
            <p className="mt-2 text-xs text-muted-foreground">
              Keep your case ID safe. You may need it when following up with authorities.
            </p>
          </motion.div>
        ) : (
          <>
            {/* Error / Offline state */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                role="alert"
                className={`glass-card p-4 text-center mb-6 ${
                  backendOffline
                    ? "border-yellow-500/30 bg-yellow-500/10"
                    : "border-destructive/50 bg-destructive/5"
                }`}
              >
                {backendOffline
                  ? <WifiOff className="mx-auto mb-2 h-6 w-6 text-yellow-400" aria-hidden="true" />
                  : <AlertTriangle className="mx-auto mb-2 h-6 w-6 text-destructive" aria-hidden="true" />
                }
                <p className={`font-semibold text-sm ${backendOffline ? "text-yellow-400" : "text-destructive"}`}>
                  {backendOffline ? "Server Offline — Report Not Submitted" : "Submission Failed"}
                </p>
                <p className="text-xs text-muted-foreground mt-1">{error}</p>
                {backendOffline && (
                  <p className="mt-3 text-xs text-muted-foreground border-t border-border/30 pt-3">
                    If you need urgent help, please contact the{" "}
                    <a
                      href="https://cybercrime.gov.in"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-neon-blue underline"
                    >
                      National Cybercrime Portal
                    </a>{" "}
                    directly.
                  </p>
                )}
              </motion.div>
            )}

            <motion.form
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              onSubmit={submit}
              className="glass-card p-8 space-y-5"
              noValidate
            >
              {fields.map((f) => (
                <div key={f.name}>
                  <label
                    className="mb-1 block text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                    htmlFor={f.name}
                  >
                    {f.label}
                  </label>
                  {f.type === "select" ? (
                    <select
                      id={f.name}
                      name={f.name}
                      value={form[f.name as keyof typeof form]}
                      onChange={(e) => update(f.name, e.target.value)}
                      required
                      className="input-cyber"
                    >
                      <option value="">Select...</option>
                      {f.options?.map((o) => (
                        <option key={o} value={o}>{o}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={f.name}
                      name={f.name}
                      type={f.type}
                      value={form[f.name as keyof typeof form]}
                      onChange={(e) => update(f.name, e.target.value)}
                      required
                      className="input-cyber"
                      placeholder={f.label}
                      maxLength={FIELD_LIMITS[f.name]}
                      {...(f.name === "age"     ? { min: "1", max: "120" } : {})}
                      {...(f.name === "contact" ? { pattern: "[0-9+\\-\\s]{7,20}", placeholder: "Phone number" } : {})}
                      {...(f.name === "email"   ? { autoComplete: "email" } : {})}
                    />
                  )}
                  {/* FIX #22: Show character counts for bounded fields */}
                  {FIELD_LIMITS[f.name] && (
                    <p className="mt-0.5 text-right text-[10px] text-muted-foreground">
                      {form[f.name as keyof typeof form].length}/{FIELD_LIMITS[f.name]}
                    </p>
                  )}
                </div>
              ))}

              {/* Description */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label
                    className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                    htmlFor="description"
                  >
                    Incident Description
                  </label>
                  <span className="text-[10px] text-muted-foreground">
                    {form.description.length}/2000
                  </span>
                </div>
                <textarea
                  id="description"
                  name="description"
                  value={form.description}
                  onChange={(e) => update("description", e.target.value.substring(0, 2000))}
                  required
                  rows={4}
                  maxLength={2000}
                  className="input-cyber resize-none"
                  placeholder="Describe the incident in detail..."
                />
              </div>

              {/* Consent */}
              <div className="flex items-start gap-2 py-2">
                <input
                  type="checkbox"
                  id="consent"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                  className="mt-0.5 rounded border-border bg-secondary text-neon-purple focus:ring-neon-purple"
                  required
                />
                <label htmlFor="consent" className="text-xs text-muted-foreground leading-normal">
                  I agree to the{" "}
                  <Link to="/privacy" className="text-neon-purple underline hover:text-neon-purple/80">
                    Privacy Policy
                  </Link>{" "}
                  and consent to my details being stored securely for digital evidence purposes.
                </label>
              </div>

              <button
                type="submit"
                disabled={loading}
                aria-busy={loading}
                className="btn-glow flex w-full items-center justify-center gap-2 font-display text-sm tracking-wide text-primary-foreground disabled:opacity-40"
              >
                {loading
                  ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  : <Send className="h-4 w-4" aria-hidden="true" />
                }
                {loading ? "Submitting..." : "Submit Report"}
              </button>
            </motion.form>
          </>
        )}
      </main>
      <Footer />
    </div>
  );
};

export default ReportPage;
