import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { ShieldAlert, ArrowLeft } from "lucide-react";
import Navbar from "@/components/Navbar";
import AnimatedBackground from "@/components/AnimatedBackground";
import GlowCursor from "@/components/GlowCursor";
import Footer from "@/components/Footer";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="relative min-h-screen cyber-grid flex flex-col justify-between">
      <AnimatedBackground />
      <GlowCursor />
      <Navbar />
      
      <main className="flex-1 flex items-center justify-center px-4 pt-24 pb-12 relative z-10">
        <div className="glass-card neon-border max-w-md w-full p-8 text-center">
          <ShieldAlert className="mx-auto mb-4 h-16 w-16 text-neon-pink animate-bounce" />
          <h1 className="font-display text-5xl font-extrabold tracking-wider text-neon-pink mb-2">404</h1>
          <h2 className="font-display text-base font-bold uppercase tracking-wider mb-2">Page Offline or Non-Existent</h2>
          <p className="text-xs text-muted-foreground mb-6 leading-relaxed">
            The database coordinate <code className="font-mono text-neon-blue px-1.5 py-0.5 rounded bg-secondary/50">{location.pathname}</code> could not be located. It may have been relocated or restricted.
          </p>
          <Link
            to="/"
            className="btn-glow inline-flex w-full items-center justify-center gap-2 font-display text-xs tracking-wider uppercase text-primary-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            Return to HQ
          </Link>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default NotFound;
