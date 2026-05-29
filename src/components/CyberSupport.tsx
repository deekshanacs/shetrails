import { useState } from "react";
import { Shield, ExternalLink, Phone, HelpCircle, AlertTriangle, FileText } from "lucide-react";
import { motion } from "framer-motion";

const faqs = [
  {
    q: "What constitutes image-based abuse?",
    a: "It is the non-consensual creation, modification, or distribution of personal images, including deepfakes, face-swaps, and morphed explicit pictures, intended to harass, intimidate, or blackmail victims."
  },
  {
    q: "How does the SHE-GUARD AI evidence report help me?",
    a: "Our report conducts pixel-level analyses (ELA and noise checks) and compiles software signatures. You can download this report and submit it directly as supporting digital evidence when filing cyber complaints."
  },
  {
    q: "Will my uploaded images be public?",
    a: "Absolutely not. Uploaded images are processed in-memory solely for forensic calculations and are never stored or exposed. Your confidentiality and privacy are legally protected."
  }
];

const helplines = [
  { name: "National Cyber Crime Helpline", number: "1930", desc: "Available 24/7 for immediate assistance (India)" },
  { name: "NCW Women Helpline", number: "7827170170", desc: "National Commission for Women Support Cell" },
  { name: "National Emergency Number", number: "112", desc: "Single emergency response system" }
];

const CyberSupport = () => {
  const [activeTab, setActiveTab] = useState<"helpline" | "faq">("helpline");

  return (
    <section className="px-4 py-24">
      <div className="mx-auto max-w-4xl">
        <div className="mb-12 text-center">
          <h2 className="mb-4 font-display text-3xl font-bold tracking-wide sm:text-4xl">
            <span className="gradient-text-accent">Support & Help Resources</span>
          </h2>
          <p className="mx-auto max-w-xl text-muted-foreground text-sm">
            Empowering victims of online abuse with immediate resources, safety FAQs, and official channels.
          </p>
        </div>

        {/* Tab Buttons */}
        <div className="mb-8 flex justify-center gap-4">
          <button
            onClick={() => setActiveTab("helpline")}
            className={`flex items-center gap-2 rounded-lg px-5 py-2.5 font-display text-xs font-semibold tracking-wider uppercase transition-all duration-300 ${
              activeTab === "helpline" 
                ? "bg-neon-purple/20 text-neon-purple border border-neon-purple/50 shadow-[0_0_10px_rgba(168,85,247,0.2)]" 
                : "border border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <Phone className="h-4 w-4" />
            Crisis Helplines
          </button>
          <button
            onClick={() => setActiveTab("faq")}
            className={`flex items-center gap-2 rounded-lg px-5 py-2.5 font-display text-xs font-semibold tracking-wider uppercase transition-all duration-300 ${
              activeTab === "faq" 
                ? "bg-neon-blue/20 text-neon-blue border border-neon-blue/50 shadow-[0_0_10px_rgba(6,182,212,0.2)]" 
                : "border border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <HelpCircle className="h-4 w-4" />
            FAQs & Guidance
          </button>
        </div>

        {/* Content Area */}
        <div className="glass-card neon-border overflow-hidden p-8">
          {activeTab === "helpline" ? (
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              <div className="mb-6 flex items-center gap-3 border-b border-border/40 pb-4">
                <AlertTriangle className="h-5 w-5 text-neon-pink" />
                <h3 className="font-display text-sm font-bold uppercase tracking-wider">Official Crime Help Cells</h3>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                {helplines.map((h) => (
                  <div key={h.name} className="flex flex-col justify-between rounded-lg border border-border/60 bg-secondary/20 p-4 text-center">
                    <div>
                      <p className="font-display text-xs font-bold uppercase tracking-wider text-foreground mb-1">{h.name}</p>
                      <p className="text-[11px] text-muted-foreground mb-3">{h.desc}</p>
                    </div>
                    <a
                      href={`tel:${h.number}`}
                      className="mt-auto block w-full rounded border border-neon-pink/40 bg-neon-pink/5 py-1.5 font-display text-xs font-semibold text-neon-pink transition-all hover:bg-neon-pink/15 hover:shadow-[0_0_10px_rgba(236,72,153,0.3)]"
                    >
                      Call {h.number}
                    </a>
                  </div>
                ))}
              </div>
              <div className="mt-6 flex flex-col items-center justify-center gap-4 rounded-lg bg-secondary/10 border border-border p-6 text-center">
                <Shield className="h-10 w-10 text-neon-purple" />
                <div>
                  <h4 className="font-display text-sm font-semibold mb-1">Official Complaint Portal</h4>
                  <p className="text-xs text-muted-foreground max-w-lg mb-3">
                    If you are a victim of cyber abuse or morphing, we strongly advise filing a formal complaint on the government portal. Your reports will be forwarded to police agencies.
                  </p>
                </div>
                <a
                  href="https://cybercrime.gov.in"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-glow inline-flex items-center gap-2 font-display text-xs tracking-wider uppercase text-primary-foreground"
                >
                  Report to Cybercrime Portal
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>
            </motion.div>
          ) : (
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              <div className="mb-6 flex items-center gap-3 border-b border-border/40 pb-4">
                <FileText className="h-5 w-5 text-neon-blue" />
                <h3 className="font-display text-sm font-bold uppercase tracking-wider">Frequently Asked Questions</h3>
              </div>
              <div className="space-y-4">
                {faqs.map((f, idx) => (
                  <div key={idx} className="rounded-lg border border-border/40 bg-secondary/15 p-5">
                    <p className="font-display text-xs font-bold uppercase tracking-wider text-neon-blue mb-2">Q: {f.q}</p>
                    <p className="text-xs text-muted-foreground leading-relaxed">A: {f.a}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </section>
  );
};

export default CyberSupport;
