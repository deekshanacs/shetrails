import { motion } from "framer-motion";
import { Shield, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

const HeroSection = () => {
  return (
    <section className="relative flex min-h-screen items-center justify-center overflow-hidden px-4">
      {/* Gradient orbs */}
      <div className="pointer-events-none absolute left-1/4 top-1/4 h-96 w-96 rounded-full bg-neon-purple/10 blur-[120px]" />
      <div className="pointer-events-none absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-neon-blue/10 blur-[120px]" />
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-64 w-64 -translate-x-1/2 -translate-y-1/2 rounded-full bg-neon-pink/5 blur-[100px]" />

      <div className="relative z-10 mx-auto max-w-5xl text-center">
        {/* Animated shield */}
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ duration: 1, type: "spring", stiffness: 100 }}
          className="mx-auto mb-8 flex h-28 w-28 items-center justify-center rounded-full neon-border"
        >
          <Shield className="h-14 w-14 text-neon-purple" />
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="mb-6 font-display text-5xl font-black tracking-wider sm:text-7xl"
        >
          <span className="gradient-text">SHE-GUARD</span>{" "}
          <span className="glow-text-blue text-neon-blue">AI</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="mx-auto mb-10 max-w-2xl font-body text-lg text-muted-foreground sm:text-xl"
        >
          AI Cyber Forensic System Protecting Women from Deepfake and Image Morphing Abuse
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.8 }}
          className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center"
        >
          <Link to="/upload" className="btn-glow flex items-center gap-2 font-display text-sm tracking-wide text-primary-foreground">
            Start Analysis
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/report"
            className="rounded-lg border border-border px-8 py-3 font-display text-sm tracking-wide text-foreground transition-all duration-300 hover:border-neon-purple/50 hover:shadow-[var(--glow-purple)]"
          >
            Report Incident
          </Link>
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
