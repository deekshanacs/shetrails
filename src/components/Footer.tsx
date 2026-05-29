import { Shield } from "lucide-react";

const Footer = () => (
  <footer className="border-t border-border px-4 py-12">
    <div className="mx-auto flex max-w-6xl flex-col items-center gap-4 text-center">
      <div className="flex items-center gap-2">
        <Shield className="h-5 w-5 text-neon-purple" />
        <span className="font-display text-sm font-bold tracking-wider gradient-text">SHE-GUARD AI</span>
      </div>
      <p className="max-w-md text-sm text-muted-foreground">
        Empowering women against digital abuse through AI.
      </p>
      <p className="text-xs text-muted-foreground/50">
        © {new Date().getFullYear()} SHE-GUARD AI. All rights reserved.
      </p>
    </div>
  </footer>
);

export default Footer;
