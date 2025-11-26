"use client";
import { useState, useEffect, useRef } from "react";
import Image from "next/image";

// Define Types
type ChatMessage = {
  type: "human" | "ai";
  content: string;
};

type APIResponse = {
  history: ChatMessage[];
  latest_response?: {
    answer: string;
    thoughts: string;
    validation: string;
    source: string;
  };
};

export default function MedicalPage() {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [latestResponse, setLatestResponse] =
    useState<APIResponse["latest_response"]>(undefined);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [attachedFile, setAttachedFile] = useState<{
    file: File;
    type: "image" | "doc";
  } | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [lang, setLang] = useState("en-US");

  // Refs for file inputs
  const imgInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // 1. Fetch History on Load
  useEffect(() => {
    fetch("http://localhost:5000/api/medical", { credentials: "include" })
      .then((res) => res.json())
      .then((data: APIResponse) => {
        setHistory(data.history || []);
        setLatestResponse(data.latest_response);
      })
      .catch((err) => console.error("Failed to load history:", err));
  }, []);

  // 2. Scroll to bottom when history changes
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, latestResponse]);

  // 3. Handle Voice Input
  const toggleVoice = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Browser does not support speech recognition.");
      return;
    }

    if (isListening) {
      setIsListening(false); // Logic to stop handled by 'end' event usually, but simplistic here
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = lang;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setQuery(transcript);
    };

    recognition.start();
  };

  // 4. Handle Form Submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query && !attachedFile) return;

    setIsLoading(true);

    // Optimistic UI Update (optional, immediately show user query)
    setHistory((prev) => [...prev, { type: "human", content: query }]);

    const formData = new FormData();
    formData.append("query", query);
    if (attachedFile) {
      if (attachedFile.type === "image")
        formData.append("image", attachedFile.file);
      else formData.append("document", attachedFile.file);
    }

    try {
      const res = await fetch("http://localhost:5000/api/medical", {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      const data: APIResponse = await res.json();

      // Update State with real data from backend
      if (data.history) setHistory(data.history);
      if (data.latest_response) setLatestResponse(data.latest_response);

      setQuery("");
      setAttachedFile(null);
    } catch (err) {
      console.error("Error submitting query:", err);
      // Ideally show error toast here
    } finally {
      setIsLoading(false);
    }
  };

  // 5. Clear Chat
  const clearChat = async () => {
    await fetch("http://localhost:5000/api/medical/clear", {
      method: "POST",
      credentials: "include",
    });
    setHistory([]);
    setLatestResponse(undefined);
  };

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-32">
      {/* HEADER / WELCOME */}
      {history.length === 0 && (
        <div className="text-center py-10">
          <i className="fa-solid fa-suitcase-medical text-blue-500 dark:text-blue-400 text-6xl mb-4 animate-float"></i>
          <h2 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">
            Medical Assistant
          </h2>
          <p className="text-lg text-gray-500 dark:text-gray-400">
            Ask a medical question, upload a medical image, or document.
          </p>
        </div>
      )}

      {/* CHAT HISTORY */}
      <div className="space-y-6 max-w-5xl mx-auto">
        {history.map((msg, idx) => (
          <div
            key={idx}
            className={`flex w-full ${
              msg.type === "human" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-2xl p-4 rounded-xl shadow-md ${
                msg.type === "human"
                  ? "bg-blue-500 text-white"
                  : "bg-white dark:bg-slate-700 border dark:border-slate-600 text-gray-800 dark:text-white"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}

        {/* LATEST AI RESPONSE (Detailed View) */}
        {latestResponse && latestResponse.answer && (
          <div className="w-full max-w-2xl space-y-4 animate-fade-in">
            {/* Details / Thoughts Accordion */}
            <details className="w-full group">
              <summary className="flex items-center text-sm text-gray-500 dark:text-gray-400 cursor-pointer mb-2 ml-1 list-none">
                <i
                  className={`fas fa-atom mr-2 text-blue-500 group-open:rotate-180 transition-transform`}
                ></i>
                <span>Thinking Process & Sources...</span>
              </summary>
              <div className="bg-gray-100 dark:bg-slate-800 border dark:border-slate-700 rounded-xl p-4 space-y-3 mt-2">
                {latestResponse.source && (
                  <div>
                    <span className="font-semibold text-gray-700 dark:text-gray-300">
                      Source:
                    </span>{" "}
                    <span className="text-sm">{latestResponse.source}</span>
                  </div>
                )}
                {latestResponse.validation && (
                  <div>
                    <span className="font-semibold text-gray-700 dark:text-gray-300">
                      Validation:
                    </span>{" "}
                    <span className="text-sm">{latestResponse.validation}</span>
                  </div>
                )}
                {latestResponse.thoughts && (
                  <div className="text-xs bg-gray-200 dark:bg-slate-900 p-2 rounded whitespace-pre-wrap font-mono">
                    {latestResponse.thoughts}
                  </div>
                )}
              </div>
            </details>

            {/* The Main Answer */}
            <div className="bg-white dark:bg-slate-700 border dark:border-slate-600 rounded-xl shadow-md p-5">
              <div className="flex items-center mb-3">
                <i className="fa-solid fa-user-doctor text-blue-500 mr-2"></i>
                <span className="font-bold text-gray-900 dark:text-white">
                  AI Response
                </span>
              </div>
              <div
                className="prose dark:prose-invert max-w-none text-gray-800 dark:text-white whitespace-pre-line"
                dangerouslySetInnerHTML={{ __html: latestResponse.answer }}
              ></div>
            </div>
          </div>
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* FOOTER INPUT AREA */}
      <footer className="fixed bottom-0 left-0 w-full z-20 bg-white/90 dark:bg-slate-900/90 backdrop-blur border-t dark:border-slate-700">
        <div className="max-w-5xl mx-auto px-4 py-3 relative">
          {/* Upload Menu */}
          {showUploadMenu && (
            <div className="absolute bottom-20 left-4 w-52 bg-white dark:bg-slate-700 border dark:border-slate-600 rounded-lg shadow-xl py-2 z-30">
              <button
                onClick={() => {
                  imgInputRef.current?.click();
                  setShowUploadMenu(false);
                }}
                className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-slate-600 flex items-center"
              >
                <i className="fas fa-image mr-3 text-blue-500"></i> Upload Image
              </button>
              <button
                onClick={() => {
                  docInputRef.current?.click();
                  setShowUploadMenu(false);
                }}
                className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-slate-600 flex items-center"
              >
                <i className="fas fa-file-alt mr-3 text-green-500"></i> Upload
                Document
              </button>
            </div>
          )}

          {/* Hidden Inputs */}
          <input
            type="file"
            ref={imgInputRef}
            accept="image/*"
            className="hidden"
            title="Upload image"
            onChange={(e) => {
              if (e.target.files?.[0])
                setAttachedFile({ file: e.target.files[0], type: "image" });
            }}
          />
          <input
            type="file"
            ref={docInputRef}
            accept=".pdf,.txt,.docx"
            className="hidden"
            title="Upload document"
            onChange={(e) => {
              if (e.target.files?.[0])
                setAttachedFile({ file: e.target.files[0], type: "doc" });
            }}
          />

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="relative flex items-center">
            {/* Plus Button */}
            <button
              type="button"
              onClick={() => setShowUploadMenu(!showUploadMenu)}
              className={`absolute left-4 z-10 p-2 rounded-full transition-colors ${
                attachedFile
                  ? "text-green-500"
                  : "text-gray-400 hover:text-blue-500"
              }`}
              aria-label={
                attachedFile
                  ? `Attached: ${attachedFile.file.name}`
                  : "Attach a file"
              }
              title={
                attachedFile
                  ? `Attached: ${attachedFile.file.name}`
                  : "Attach a file"
              }
            >
              <i
                className={`fas ${
                  attachedFile ? "fa-check-circle" : "fa-plus"
                } text-xl`}
              ></i>
            </button>

            {/* Text Field */}
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                attachedFile
                  ? `Attached: ${attachedFile.file.name}`
                  : "Ask a medical question..."
              }
              className="w-full pl-14 pr-36 py-4 rounded-full bg-gray-100 dark:bg-slate-800 border-transparent focus:border-blue-500 focus:bg-white dark:focus:bg-slate-700 focus:ring-0 transition-all shadow-sm outline-none"
            />

            {/* Right Controls */}
            <div className="absolute right-2 flex items-center space-x-2">
              <select
                value={lang}
                onChange={(e) => setLang(e.target.value)}
                aria-label="Language"
                title="Language"
                className="bg-transparent text-xs font-bold text-gray-500 dark:text-gray-400 uppercase cursor-pointer outline-none border-none focus:ring-0"
              >
                <option value="en-US">EN</option>
                <option value="ms-MY">MS</option>
                <option value="ar-SA">AR</option>
              </select>

              <div className="h-4 w-px bg-gray-300 dark:bg-slate-600"></div>

              <button
                type="button"
                onClick={toggleVoice}
                className={`p-2 rounded-full transition-colors ${
                  isListening
                    ? "text-red-500 animate-pulse"
                    : "text-gray-400 hover:text-blue-500"
                }`}
                aria-label={
                  isListening ? "Stop voice input" : "Start voice input"
                }
                title={isListening ? "Stop voice input" : "Start voice input"}
              >
                <i className="fas fa-microphone text-lg"></i>
              </button>

              <button
                type="submit"
                disabled={isLoading}
                className="bg-blue-500 hover:bg-blue-600 text-white p-3 rounded-full shadow-lg transition-transform transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Send message"
                title="Send message"
              >
                {isLoading ? (
                  <i className="fas fa-spinner fa-spin"></i>
                ) : (
                  <i className="fas fa-paper-plane"></i>
                )}
              </button>
            </div>
          </form>

          {/* Clear Chat Button (Small) */}
          <button
            onClick={clearChat}
            className="absolute -top-10 right-0 text-xs text-red-400 hover:text-red-600 underline"
          >
            Clear Chat History
          </button>
        </div>
      </footer>
    </main>
  );
}
