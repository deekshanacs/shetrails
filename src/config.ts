const productionApiBaseUrl = "https://sheguard-api.onrender.com";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "http://localhost:7860" : productionApiBaseUrl);

