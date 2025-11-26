"use client";
import "./globals.css";
import { Inter } from "next/font/google";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  // Generate stars on load
  useEffect(() => {
    const container = document.getElementById("stars-container");
    if (container && container.childElementCount === 0) {
      for (let i = 0; i < 50; i++) {
        const star = document.createElement("div");
        star.className = "star";
        star.style.top = `${Math.random() * 100}%`;
        star.style.left = `${Math.random() * 100}%`;
        star.style.width = `${Math.random() * 3}px`;
        star.style.height = star.style.width;
        star.style.setProperty("--duration", `${Math.random() * 3 + 2}s`);
        star.style.setProperty("--opacity", `${Math.random()}`);
        container.appendChild(star);
      }
    }
  }, []);

  return (
    <html lang="en" className="dark">
      <head>
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
        />
      </head>
      <body
        className={`${inter.className} bg-cosmic-900 text-white min-h-screen overflow-x-hidden selection:bg-cosmic-glow selection:text-white`}
      >
        {/* Star Background */}
        <div id="stars-container" className="stars-container"></div>

        {/* Top Horizon Glow */}
        <div className="black-hole-horizon"></div>

        {/* NAVIGATION - Floating Pill Style */}
        <nav className="fixed top-6 left-0 right-0 z-50 flex justify-center">
          <div className="glass-card px-8 py-3 rounded-full flex items-center space-x-8 bg-[#0300145e] border-[#7042f861]">
            {/* Logo */}
            <Link
              href="/"
              className="font-bold text-xl tracking-wider flex items-center gap-2"
            >
              {/* Simple Text Logo as per portfolio style */}
              <span className="text-gray-300">MultiDom</span>
              <span className="text-cosmic-glow">RAG</span>
            </Link>

            {/* Links */}
            <div className="hidden md:flex items-center space-x-1 bg-[#0f0c29] px-2 py-1 rounded-full border border-[#7042f861]">
              <NavLink href="/" current={pathname} label="Home" />
              <NavLink href="/medical" current={pathname} label="Medical" />
              <NavLink href="/islamic" current={pathname} label="Islamic" />
              <NavLink href="/insurance" current={pathname} label="Insurance" />
              <NavLink href="/about" current={pathname} label="About" />
            </div>

            {/* Socials (Simulated) */}
            <div className="flex gap-4 text-gray-400">
              <i className="fab fa-github hover:text-white cursor-pointer transition"></i>
              <i className="fab fa-linkedin hover:text-white cursor-pointer transition"></i>
            </div>
          </div>
        </nav>

        <div className="pt-32">{children}</div>

        {/* FOOTER */}
        <footer className="relative z-10 border-t border-[#2A0E61] mt-20 bg-[#030014]">
          <div className="max-w-7xl mx-auto py-8 px-4 text-center">
            <p className="text-gray-500 text-sm">
              &copy; 2025 Izzmir's MultiDom RAG. Built with{" "}
              <span className="text-cosmic-glow">Cosmic Intelligence</span>.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}

// Helper Component for Links
function NavLink({
  href,
  current,
  label,
}: {
  href: string;
  current: string;
  label: string;
}) {
  const isActive = current === href;
  return (
    <Link
      href={href}
      className={`px-4 py-1 rounded-full text-sm font-medium transition-all duration-300 ${
        isActive
          ? "bg-cosmic-glow text-white shadow-[0_0_10px_#6536F0]"
          : "text-gray-300 hover:text-white hover:bg-[#2A0E61]"
      }`}
    >
      {label}
    </Link>
  );
}
