import { BarChart3, Cpu, FolderOpen, Mic, Terminal } from "lucide-react";
import { useState, type ReactNode } from "react";
import { StatusPill, Tabs } from "../components/ui";
import { BatchTranscription } from "../features/batch/BatchTranscription";
import { Benchmark } from "../features/benchmark/Benchmark";
import { Diagnostics } from "../features/diagnostics/Diagnostics";
import { LiveMicAudio } from "../features/live/LiveMicAudio";
import { Training } from "../features/training/Training";
import type { StudioTab } from "../types/asr";

const tabItems: Array<{ value: StudioTab; label: string; icon: ReactNode }> = [
  { value: "live", label: "Live Mic & Audio", icon: <Mic size={16} /> },
  { value: "batch", label: "Batch", icon: <FolderOpen size={16} /> },
  { value: "benchmark", label: "Benchmark", icon: <BarChart3 size={16} /> },
  { value: "training", label: "Training", icon: <Terminal size={16} /> },
  { value: "diagnostics", label: "Diagnostics", icon: <Cpu size={16} /> }
];

export function AppShell() {
  const [tab, setTab] = useState<StudioTab>("live");

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Bangla & English ASR Studio</h1>
          <p>React + FastAPI workstation console for Whisper inference, evaluation, and training.</p>
        </div>
        <div className="topbar-pills">
          <StatusPill tone="good">Local API</StatusPill>
          <StatusPill>Whisper</StatusPill>
          <StatusPill>LoRA Ready</StatusPill>
        </div>
      </header>
      <Tabs label="Studio sections">
        {tabItems.map((item) => (
          <button key={item.value} className={`tab-btn ${tab === item.value ? "active" : ""}`} onClick={() => setTab(item.value)}>
            {item.icon}
            {item.label}
          </button>
        ))}
      </Tabs>
      {tab === "live" && <LiveMicAudio />}
      {tab === "batch" && <BatchTranscription />}
      {tab === "benchmark" && <Benchmark />}
      {tab === "training" && <Training />}
      {tab === "diagnostics" && <Diagnostics />}
    </main>
  );
}
