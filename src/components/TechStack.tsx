import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Brain, Eye, ShieldCheck, Fingerprint } from "lucide-react";

const techs = [
  { 
    icon: Brain, 
    title: "Vite + React & TS", 
    desc: "Blazing fast single-page app utilizing TypeScript, Framer Motion, and TailwindCSS for high-performance visual forensic rendering." 
  },
  { 
    icon: Eye, 
    title: "OpenCV & NumPy", 
    desc: "Powers the algorithmic forensic image checkers, including Haar Cascades, Sobel edge mapping, and standard deviation calculations." 
  },
  { 
    icon: ShieldCheck, 
    title: "FastAPI Backend", 
    desc: "Asynchronous Python server conducting Error Level Analysis, parsing metadata, and exposing incident data endpoints." 
  },
  { 
    icon: Fingerprint, 
    title: "SQLite & SQLAlchemy", 
    desc: "Secure relational storage tracking incident logs, metadata logs, and cases history linked with cryptographic Case IDs." 
  },
];

const TechStack = () => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="px-4 py-24">
      <div className="mx-auto max-w-6xl">
        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }}
          className="mb-16 text-center font-display text-3xl font-bold tracking-wide sm:text-4xl"
        >
          Powered By <span className="gradient-text">Advanced Technology</span>
        </motion.h2>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {techs.map((t, i) => (
            <motion.div
              key={t.title}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ delay: 0.15 * i, duration: 0.5 }}
              className="glass-card-hover flex flex-col items-center p-8 text-center"
            >
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-neon-blue/10">
                <t.icon className="h-8 w-8 text-neon-blue" />
              </div>
              <h3 className="mb-2 font-display text-sm font-semibold tracking-wider uppercase text-foreground">{t.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{t.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TechStack;
