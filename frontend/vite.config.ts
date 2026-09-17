import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Inside Docker the backend is reachable as http://backend:8000; on a bare host
// it is http://127.0.0.1:8000. VITE_PROXY_TARGET lets both work unchanged.
const proxyTarget = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Poll-based watching is required for bind mounts on macOS/Windows.
    watch: process.env.CHOKIDAR_USEPOLLING === "true" ? { usePolling: true } : undefined,
    proxy: {
      "/api": {
        target: proxyTarget,
        changeOrigin: true,
        // Job progress is Server-Sent Events — never buffer it.
        timeout: 0,
        proxyTimeout: 0
      }
    }
  }
});

