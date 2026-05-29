import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AnimatePresence } from "framer-motion";
import PageTransition from "@/components/PageTransition";
import Index from "./pages/Index.tsx";
import UploadPage from "./pages/UploadPage.tsx";
import ReportPage from "./pages/ReportPage.tsx";
import EvidenceReport from "./pages/EvidenceReport.tsx";
import DashboardPage from "./pages/DashboardPage.tsx";
import NotFound from "./pages/NotFound.tsx";

import PrivacyPage from "./pages/PrivacyPage.tsx";

const queryClient = new QueryClient();

const AnimatedRoutes = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageTransition><Index /></PageTransition>} />
        <Route path="/upload" element={<PageTransition><UploadPage /></PageTransition>} />
        <Route path="/report" element={<PageTransition><ReportPage /></PageTransition>} />
        <Route path="/evidence" element={<PageTransition><EvidenceReport /></PageTransition>} />
        <Route path="/evidence-report" element={<PageTransition><EvidenceReport /></PageTransition>} />
        <Route path="/dashboard" element={<PageTransition><DashboardPage /></PageTransition>} />
        <Route path="/privacy" element={<PageTransition><PrivacyPage /></PageTransition>} />
        <Route path="*" element={<PageTransition><NotFound /></PageTransition>} />
      </Routes>
    </AnimatePresence>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AnimatedRoutes />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
