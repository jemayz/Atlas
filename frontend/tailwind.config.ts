import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Portfolio Colors extracted from image
        cosmic: {
          900: "#030014", // Deepest space background
          800: "#0F0C29", // Slightly lighter
          700: "#2A0E61", // Deep Purple
          glow: "#6536F0", // The bright neon violet
          accent: "#AD26FF", // Pinkish purple accent
        },
        glass: {
          100: "rgba(255, 255, 255, 0.05)",
          200: "rgba(255, 255, 255, 0.1)",
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "cosmic-gradient":
          "linear-gradient(to bottom, #030014 0%, #2A0E61 50%, #030014 100%)",
      },
      animation: {
        "spin-slow": "spin 20s linear infinite",
        "pulse-glow": "pulse-glow 4s ease-in-out infinite",
      },
      keyframes: {
        "pulse-glow": {
          "0%, 100%": { opacity: "0.6", transform: "scale(1)" },
          "50%": { opacity: "1", transform: "scale(1.1)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
