import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Upload, Brain, Search, FileText, ShieldCheck } from "lucide-react";

const steps = [
  { icon: Upload, title: "Upload Image", desc: "Submit the suspected image securely via our dashboard." },
  { icon: Brain, title: "AI Forensic Analysis", desc: "Digital verification pipelines audit local pixels for metadata changes." },
  { icon: Search, title: "Artifact Detection", desc: "Identify ELA anomalies, noise inconsistency, and face-swap signatures." },
  { icon: FileText, title: "Evidence Report", desc: "Generate a cryptographically stamped forensic evidence PDF/Text report." },
  { icon: ShieldCheck, title: "Cybercrime Portal", desc: "Guided path to file reports immediately with law enforcement." },
];

const HowItWorks = () => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="relative px-4 py-24 overflow-hidden">
      <div className="mx-auto max-w-5xl">
        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }}
          className="mb-20 text-center font-display text-3xl font-bold tracking-wide sm:text-4xl"
        >
          How <span className="gradient-text">It Works</span>
        </motion.h2>

        <div className="relative">
          {/* Central Vertical Timeline Line */}
          <div className="absolute left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 bg-gradient-to-b from-neon-purple via-neon-blue to-neon-purple/20 hidden md:block"></div>

          <div className="space-y-12 md:space-y-0 relative">
            {steps.map((step, i) => {
              const isEven = i % 2 === 0;
              return (
                <div key={step.title} className="flex flex-col md:flex-row items-center md:justify-between w-full md:mb-16 last:mb-0">
                  {/* Left Side (Odd items show card on left, Even items show empty space on desktop) */}
                  <div className={`w-full md:w-[45%] ${isEven ? "md:text-right md:order-1" : "md:order-3 md:invisible h-0 md:h-auto"}`}>
                    {isEven && (
                      <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        animate={inView ? { opacity: 1, x: 0 } : {}}
                        transition={{ delay: 0.15 * i, duration: 0.6 }}
                        className="glass-card-hover p-6 text-left"
                      >
                        <div className="flex items-center gap-4 mb-3 md:flex-row-reverse md:justify-between">
                          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-neon-blue/10">
                            <step.icon className="h-6 w-6 text-neon-blue" />
                          </div>
                          <span className="font-display text-2xl font-bold text-neon-purple/30">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                        </div>
                        <h3 className="font-display text-base font-semibold tracking-wide mb-1">{step.title}</h3>
                        <p className="text-sm text-muted-foreground">{step.desc}</p>
                      </motion.div>
                    )}
                  </div>

                  {/* Central Node Circle (Desktop only) */}
                  <div className="absolute left-1/2 -translate-x-1/2 h-8 w-8 rounded-full border border-neon-purple bg-cyber-dark z-10 flex items-center justify-center shadow-[0_0_10px_rgba(168,85,247,0.5)] hidden md:flex md:order-2">
                    <div className="h-2.5 w-2.5 rounded-full bg-neon-blue animate-pulse"></div>
                  </div>

                  {/* Right Side (Even items show empty space, Odd items show card on right on desktop) */}
                  <div className={`w-full md:w-[45%] ${!isEven ? "md:order-3" : "md:order-1 md:invisible h-0 md:h-auto"}`}>
                    {!isEven && (
                      <motion.div
                        initial={{ opacity: 0, x: 50 }}
                        animate={inView ? { opacity: 1, x: 0 } : {}}
                        transition={{ delay: 0.15 * i, duration: 0.6 }}
                        className="glass-card-hover p-6"
                      >
                        <div className="flex items-center gap-4 mb-3 justify-between">
                          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-neon-purple/10">
                            <step.icon className="h-6 w-6 text-neon-purple" />
                          </div>
                          <span className="font-display text-2xl font-bold text-neon-blue/30">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                        </div>
                        <h3 className="font-display text-base font-semibold tracking-wide mb-1">{step.title}</h3>
                        <p className="text-sm text-muted-foreground">{step.desc}</p>
                      </motion.div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
