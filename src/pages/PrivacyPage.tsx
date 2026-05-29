import { motion } from "framer-motion";
import { Shield, Eye, Lock, FileText, ArrowLeft } from "lucide-react";
import Navbar from "@/components/Navbar";
import AnimatedBackground from "@/components/AnimatedBackground";
import GlowCursor from "@/components/GlowCursor";
import Footer from "@/components/Footer";
import { Link } from "react-router-dom";

const PrivacyPage = () => {
  return (
    <div className="relative min-h-screen cyber-grid">
      <AnimatedBackground />
      <GlowCursor />
      <Navbar />
      <main className="mx-auto max-w-3xl px-4 pb-24 pt-28">
        <Link to="/" className="inline-flex items-center gap-2 text-xs uppercase tracking-widest text-muted-foreground hover:text-foreground mb-6 transition-colors">
          <ArrowLeft className="h-4 w-4" />
          Back to home
        </Link>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 text-center"
        >
          <div className="mb-4 flex justify-center">
            <Shield className="h-10 w-10 text-neon-purple animate-pulse" />
          </div>
          <h1 className="font-display text-2xl font-bold tracking-wide sm:text-3xl">
            <span className="gradient-text">Privacy Policy & Data Security</span>
          </h1>
          <p className="mt-2 text-muted-foreground text-xs uppercase tracking-widest">Effective Date: May 2026</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card neon-border p-8 space-y-6"
        >
          <p className="text-sm text-muted-foreground leading-relaxed">
            At SHE-GUARD AI, we are committed to providing the highest standard of security and confidentiality for victims of image-based abuse. We process sensitive files and personal details under strict protocols.
          </p>

          <div className="space-y-4">
            <Section icon={Eye} title="1. Volatile In-Memory Image Analysis">
              <p className="text-xs text-muted-foreground leading-relaxed">
                Any image you upload to the Forensic Analyze portal is processed **strictly in-memory** on our secure backend server. We do not store, copy, share, or archive your files. The image bytes are discarded immediately after the mathematical calculations (ELA and noise checks) are returned.
              </p>
            </Section>

            <Section icon={Lock} title="2. Secure Encryption & Data Isolation">
              <p className="text-xs text-muted-foreground leading-relaxed">
                All submitted reports (including names, contact numbers, and case descriptions) are stored securely. Access to these logs is restricted behind security gates and is intended only for verified law enforcement officers and forensic experts.
              </p>
            </Section>

            <Section icon={FileText} title="3. Data Retention & Erasure">
              <p className="text-xs text-muted-foreground leading-relaxed">
                Incident logs are retained solely to assist in cyber forensics investigations. Victims can request the deletion of their personal case logs from the active database at any time by contacting our support officers.
              </p>
            </Section>
          </div>

          <div className="border-t border-border/40 pt-6 text-center">
            <p className="text-[11px] text-muted-foreground italic">
              SHE-GUARD AI complies with national data protection regulations. For questions regarding data processing, please contact support at security@sheguard.ai.
            </p>
          </div>
        </motion.div>
      </main>
      <Footer />
    </div>
  );
};

const Section = ({ icon: Icon, title, children }: { icon: React.ElementType; title: string; children: React.ReactNode }) => (
  <div className="space-y-2">
    <div className="flex items-center gap-2">
      <Icon className="h-4.5 w-4.5 text-neon-blue" />
      <h3 className="font-display text-xs font-semibold uppercase tracking-wider text-foreground">{title}</h3>
    </div>
    <div className="pl-6">{children}</div>
  </div>
);

export default PrivacyPage;
