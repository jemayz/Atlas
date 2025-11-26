"use client";
import Link from "next/link";
import { useState } from "react";

export default function Home() {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  return (
    <main className="flex flex-col items-center w-full bg-cosmic-900 min-h-screen overflow-hidden">
      {/* --- STAR SHATTER BACKGROUND --- */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        {/* Stars are generated in layout.tsx or globals.css, but we can add a specific overlay here if needed */}
        <div className="stars-container"></div>
      </div>

      {/* --- VIDEO BACKGROUND HERO --- */}
      <section className="relative w-full h-screen flex flex-col items-center justify-center text-center px-4 z-10 overflow-hidden">
        {/* Video Background */}
        <div className="absolute inset-0 w-full h-full z-0">
          <video
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-full object-cover opacity-40"
          >
            <source src="public/videos/blackhole.webm" type="video/webm" />
            Your browser does not support the video tag.
          </video>
          {/* Gradient Overlay for legibility */}
          <div className="absolute inset-0 bg-gradient-to-b from-cosmic-900/80 via-transparent to-cosmic-900"></div>
        </div>

        {/* Glowing Tag */}
        <div className="relative z-20 mb-6 inline-flex items-center gap-2 px-3 py-1 rounded-md border border-[#7042f88b] bg-[rgba(8,5,30,0.6)] backdrop-blur-md animate-fade-in-up">
          <i className="fas fa-sparkles text-[#b49bff]"></i>
          <span className="text-[#b49bff] text-sm font-medium">
            Powered by Swarm Intelligence
          </span>
        </div>

        {/* Main Title */}
        <h1 className="relative z-20 text-5xl md:text-7xl font-bold text-white max-w-4xl leading-tight mb-6 drop-shadow-lg">
          Welcome to{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-cyan-500">
            Atlas
          </span>{" "}
          <br />
          The Multi-Domain RAG System
        </h1>

        {/* Subtitle */}
        <p className="relative z-20 text-gray-300 text-lg md:text-xl max-w-2xl mb-10 drop-shadow-md">
          An advanced AI orchestrator capable of handling specialized queries in
          Medical, Islamic, and Insurance domains with high-precision Swarm
          Agents.
        </p>

        {/* CTA Buttons with Dropdown */}
        <div className="relative z-20 flex gap-6 items-center">
          {/* Demo Dropdown Button */}
          <div className="relative">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="btn-cosmic px-8 py-3 rounded-lg text-white font-semibold transition-all duration-300 flex items-center gap-2"
            >
              Try Demo{" "}
              <i
                className={`fas fa-chevron-down transition-transform ${
                  isDropdownOpen ? "rotate-180" : ""
                }`}
              ></i>
            </button>

            {/* Dropdown Menu */}
            {isDropdownOpen && (
              <div className="absolute top-full left-0 mt-2 w-48 bg-[#0f0c29] border border-[#7042f861] rounded-lg shadow-[0_0_15px_rgba(112,66,248,0.5)] overflow-hidden flex flex-col z-50 animate-fade-in">
                <Link
                  href="/medical"
                  className="px-4 py-3 text-left text-gray-300 hover:text-white hover:bg-[#2A0E61] transition-colors border-b border-[#7042f82d]"
                >
                  <i className="fas fa-heart-pulse mr-2 text-red-400"></i>{" "}
                  Medical
                </Link>
                <Link
                  href="/islamic"
                  className="px-4 py-3 text-left text-gray-300 hover:text-white hover:bg-[#2A0E61] transition-colors border-b border-[#7042f82d]"
                >
                  <i className="fas fa-moon mr-2 text-emerald-400"></i> Islamic
                </Link>
                <Link
                  href="/insurance"
                  className="px-4 py-3 text-left text-gray-300 hover:text-white hover:bg-[#2A0E61] transition-colors"
                >
                  <i className="fas fa-shield-halved mr-2 text-amber-400"></i>{" "}
                  Insurance
                </Link>
              </div>
            )}
          </div>

          <Link
            href="/about"
            className="px-8 py-3 rounded-lg border border-[#7042f861] text-white hover:bg-[#7042f81f] transition-all backdrop-blur-sm"
          >
            Learn More
          </Link>
        </div>
      </section>

      {/* --- DOMAINS / SKILLS SECTION --- */}
      <section className="py-20 w-full max-w-7xl px-4 z-10 relative">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-16 text-white">
          Atlas{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-cyan-500">
            Capabilities
          </span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Domain 1: Medical */}
          <DomainCard
            href="/medical"
            icon="fa-heart-pulse"
            title="Medical AI"
            desc="Diagnostic image analysis & symptom checking using verified medical datasets."
            color="text-red-400"
            image="https://www.malaymail.com/malaymail/uploads/images/2023/09/05/144179.jpg"
          />

          {/* Domain 2: Islamic */}
          <DomainCard
            href="/islamic"
            icon="fa-moon"
            title="Islamic AI"
            desc="Fiqh and Hadith retrieval from authenticated scholarly sources."
            color="text-emerald-400"
            image="https://c7c8edde.rocketcdn.me/wp-content/uploads/mishary-alafasy-svnLIZ6jgCQ-unsplash-scaled.jpg"
          />

          {/* Domain 3: Insurance */}
          <DomainCard
            href="/insurance"
            icon="fa-shield-halved"
            title="Insurance AI"
            desc="Policy analysis and risk assessment for Etiqa Takaful products."
            color="text-amber-400"
            image="https://theedgemalaysia.com/_next/image?url=https%3A%2F%2Fassets.theedgemarkets.com%2FEtiqa_TEM1464_20230321165006_theedgemarkets.jpg&w=1920&q=75"
          />
        </div>
      </section>
    </main>
  );
}

function DomainCard({
  href,
  icon,
  title,
  desc,
  color,
  image,
}: {
  href: string;
  icon: string;
  title: string;
  desc: string;
  color: string;
  image: string;
}) {
  return (
    <Link
      href={href}
      className="glass-card rounded-2xl flex flex-col items-center text-center transition-all duration-300 hover:-translate-y-2 group cursor-pointer overflow-hidden h-full border border-[#2A0E61]"
    >
      {/* Image Banner */}
      <div className="w-full h-40 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-[#030014] to-transparent z-10"></div>
        <img
          src={image}
          alt={title}
          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500 opacity-80 group-hover:opacity-100"
        />
      </div>

      <div className="p-6 relative z-20 -mt-10 flex flex-col items-center">
        <div
          className={`w-16 h-16 rounded-full bg-[#2A0E61] border border-[#7042f861] flex items-center justify-center mb-4 group-hover:shadow-[0_0_20px_rgba(101,54,240,0.6)] transition-shadow shadow-lg`}
        >
          <i className={`fas ${icon} text-3xl ${color}`}></i>
        </div>
        <h3 className="text-2xl font-bold text-white mb-3">{title}</h3>
        <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
      </div>
    </Link>
  );
}
