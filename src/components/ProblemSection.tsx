import { motion } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef } from "react";
import { Eye, Layers, MessageCircleWarning, Lock } from "lucide-react";

const problems = [
  { icon: Eye, title: "Deepfake Images", desc: "AI-generated fake images used to harass and defame women online." },
  { icon: Layers, title: "Photo Morphing", desc: "Manipulated photos combining victims' faces with explicit content." },
  { icon: MessageCircleWarning, title: "Online Harassment", desc: "Distribution of manipulated images to intimidate and shame victims." },
  { icon: Lock, title: "Cyber Blackmail", desc: "Extortion using fabricated or altered images to exploit victims." },
];

const ProblemSection = () => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="relative px-4 py-24">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }}
          className="mb-16 text-center"
        >
          <h2 className="mb-4 font-display text-3xl font-bold tracking-wide sm:text-4xl">
            The <span className="gradient-text-accent">Growing Threat</span>
          </h2>
          <p className="mx-auto max-w-xl text-muted-foreground">
            Digital image abuse targeting women is escalating at an alarming rate. Understanding the threat is the first step toward fighting it.
          </p>
        </motion.div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4 mb-16">
          {problems.map((item, i) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 40 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.2 * i, duration: 0.6 }}
              className="glass-card-hover p-6 text-center"
            >
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-neon-purple/10">
                <item.icon className="h-7 w-7 text-neon-purple" />
              </div>
              <h3 className="mb-2 font-display text-sm font-semibold tracking-wide">{item.title}</h3>
              <p className="text-sm text-muted-foreground">{item.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Cited Statistics Grid */}
        <div className="grid gap-6 md:grid-cols-2 max-w-4xl mx-auto border-t border-border/40 pt-12">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="flex items-center gap-6 rounded-lg bg-secondary/10 border border-border/40 p-6"
          >
            <div className="font-display text-5xl font-extrabold text-neon-pink">96%</div>
            <div className="text-xs text-muted-foreground leading-relaxed">
              <p className="font-display font-semibold text-foreground uppercase tracking-wider mb-1">Targeted Demographics</p>
              of all deepfake videos and images hosted online are non-consensual sexual content, with women representing 99% of targeted individuals. <span className="text-neon-purple">(Source: Deeptrace Labs Research)</span>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="flex items-center gap-6 rounded-lg bg-secondary/10 border border-border/40 p-6"
          >
            <div className="font-display text-5xl font-extrabold text-neon-blue">460%</div>
            <div className="text-xs text-muted-foreground leading-relaxed">
              <p className="font-display font-semibold text-foreground uppercase tracking-wider mb-1">Exponential Rise</p>
              increase in reported instances of image morphing, identity spoofing, and synthetic media blackmail cases since 2022. <span className="text-neon-blue">(Source: Cyber Security Research Group)</span>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default ProblemSection;
