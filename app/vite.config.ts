import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig(({ mode }) => ({
  base: '/',
  define: {
    __KELLYCLIP_API_BASE__: mode === 'development'
      ? JSON.stringify('http://127.0.0.1:8000')
      : 'window.location.origin',
  },
  plugins: [react(), tailwindcss()],
}));
