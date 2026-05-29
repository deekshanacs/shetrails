// Mock API responses

export interface AnalysisResult {
  caseId: string;
  imageStatus: "Safe" | "Suspicious" | "Manipulated";
  confidenceScore: number;
  forensicScore: number;
  timestamp: string;
  riskLevel: "green" | "yellow" | "red";
  details: {
    faceManipulation: number;
    spliceDetection: number;
    metadataAnomaly: number;
    noiseAnalysis: number;
  };
}

export interface IncidentReport {
  caseId: string;
  name: string;
  gender: string;
  age: string;
  location: string;
  contact: string;
  description: string;
  status: string;
  submittedAt: string;
}

export const mockAnalyzeImage = (): Promise<AnalysisResult> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const statuses: AnalysisResult["imageStatus"][] = ["Safe", "Suspicious", "Manipulated"];
      const riskLevels: AnalysisResult["riskLevel"][] = ["green", "yellow", "red"];
      const idx = Math.floor(Math.random() * 3);
      resolve({
        caseId: `SG-${Date.now().toString(36).toUpperCase()}`,
        imageStatus: statuses[idx],
        confidenceScore: Math.round((70 + Math.random() * 29) * 10) / 10,
        forensicScore: Math.round((60 + Math.random() * 39) * 10) / 10,
        timestamp: new Date().toISOString(),
        riskLevel: riskLevels[idx],
        details: {
          faceManipulation: Math.round(Math.random() * 100),
          spliceDetection: Math.round(Math.random() * 100),
          metadataAnomaly: Math.round(Math.random() * 100),
          noiseAnalysis: Math.round(Math.random() * 100),
        },
      });
    }, 3000);
  });
};

export const mockReportIncident = (data: Omit<IncidentReport, "caseId" | "status" | "submittedAt">): Promise<IncidentReport> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        ...data,
        caseId: `SG-${Date.now().toString(36).toUpperCase()}`,
        status: "Filed",
        submittedAt: new Date().toISOString(),
      });
    }, 1500);
  });
};
