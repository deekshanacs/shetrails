import { Link, useLocation, useNavigate } from "react-router-dom";
import { Shield, Menu, X } from "lucide-react";
import { useState, useCallback } from "react";

const pageNavItems = [
  { label: "Analyze", path: "/upload" },
  { label: "Report", path: "/report" },
  { label: "Dashboard", path: "/dashboard" },
];

const sectionNavItems = [
  { label: "Problem", id: "problem" },
  { label: "How It Works", id: "how-it-works" },
  { label: "Tech", id: "tech-stack" },
  { label: "Support", id: "cyber-support" },
];

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const isHome = location.pathname === "/";

  const scrollToSection = useCallback((id: string) => {
    setOpen(false);
    if (!isHome) {
      navigate("/");
      setTimeout(() => {
        document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
      }, 400);
    } else {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    }
  }, [isHome, navigate]);

  return (
    <nav className="fixed left-0 right-0 top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to="/" className="flex items-center gap-2" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
          <Shield className="h-6 w-6 text-neon-purple" />
          <span className="font-display text-sm font-bold tracking-wider gradient-text">SHE-GUARD AI</span>
        </Link>

        {/* Desktop */}
        <div className="hidden items-center gap-1 sm:flex">
          {sectionNavItems.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className="rounded-lg px-3 py-2 font-body text-sm text-muted-foreground transition-colors hover:text-foreground hover:bg-neon-purple/10"
            >
              {item.label}
            </button>
          ))}
          {pageNavItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`rounded-lg px-4 py-2 font-body text-sm transition-colors ${
                location.pathname === item.path
                  ? "bg-neon-purple/10 text-neon-purple"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </div>

        {/* Mobile toggle */}
        <button className="sm:hidden text-foreground" onClick={() => setOpen(!open)}>
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="border-t border-border/50 bg-background/95 backdrop-blur-xl sm:hidden">
          {sectionNavItems.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className="block w-full text-left px-6 py-3 text-sm text-muted-foreground hover:bg-neon-purple/10"
            >
              {item.label}
            </button>
          ))}
          {pageNavItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setOpen(false)}
              className={`block px-6 py-3 text-sm ${
                location.pathname === item.path
                  ? "bg-neon-purple/10 text-neon-purple"
                  : "text-muted-foreground"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
