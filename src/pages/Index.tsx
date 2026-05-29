import AnimatedBackground from "@/components/AnimatedBackground";
import GlowCursor from "@/components/GlowCursor";
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import ProblemSection from "@/components/ProblemSection";
import HowItWorks from "@/components/HowItWorks";
import TechStack from "@/components/TechStack";
import CyberSupport from "@/components/CyberSupport";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="relative min-h-screen cyber-grid">
      <AnimatedBackground />
      <GlowCursor />
      <Navbar />
      <main>
        <HeroSection />
        <div id="problem"><ProblemSection /></div>
        <div id="how-it-works"><HowItWorks /></div>
        <div id="tech-stack"><TechStack /></div>
        <div id="cyber-support"><CyberSupport /></div>
      </main>
      <Footer />
    </div>
  );
};

export default Index;
