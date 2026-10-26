import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development the API runs separately (`repo-assist serve`); in production
// FastAPI serves the built app from the same origin, so requests use relative /api paths.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
});
