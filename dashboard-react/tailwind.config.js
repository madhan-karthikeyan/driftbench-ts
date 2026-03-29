/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0f172a',
          800: '#1e293b',
          700: '#334155',
          600: '#475569',
        },
        model: {
          naive: '#64748b',
          seasonal_naive: '#f59e0b',
          rf: '#8b5cf6',
          lgbm: '#06b6d4',
          lstm: '#ec4899',
          tsmixer: '#84cc16',
        },
        strategy: {
          no_retrain: '#ef4444',
          fixed_retrain: '#3b82f6',
          adaptive_retrain: '#22c55e',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
