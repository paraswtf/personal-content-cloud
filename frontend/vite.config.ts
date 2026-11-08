import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { apiBuildFrom } from './scripts/api-build.mjs'

/** The API build id of the backend checkout next to this frontend, '' when it is not there. */
function apiBuild(): string {
  try {
    return apiBuildFrom(import.meta.url)
  } catch {
    return ''
  }
}

export default defineConfig(({ mode }) => {
  // In dev, the API runs separately (uvicorn); BACKEND_URL (from the environment or .env) points at it.
  const env = loadEnv(mode, '.', '')
  const backend = env.BACKEND_URL || 'http://127.0.0.1:41010'
  return {
    plugins: [react(), tailwindcss()],
    define: { __API_BUILD__: JSON.stringify(apiBuild()) },
    server: {
      host: '127.0.0.1',
      proxy: { '/api': backend, '/files': backend },
    },
    build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 900 },
  }
})
