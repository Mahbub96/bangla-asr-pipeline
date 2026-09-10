import { Waves } from "lucide-react";

export function WaveformViewer({ bars, active, level }: { bars: number[]; active: boolean; level: number }) {
  if (!active && bars.length === 0) {
    return (
      <div className="waveform empty-waveform" aria-label="No audio waveform loaded">
        <div className="waveform-ruler">
          <span>0:00</span>
          <span>timeline</span>
          <span>clip end</span>
        </div>
        <div className="waveform-placeholder">
          <Waves size={28} />
          <span>Waveform appears after recording or upload</span>
        </div>
      </div>
    );
  }

  const displayBars = bars.length
    ? bars
    : Array.from({ length: 96 }, (_, index) => {
        if (!active) return 0.08;
        const phase = Math.sin(index * 0.55 + Date.now() / 130);
        return Math.max(0.05, Math.min(1, level * (0.45 + Math.abs(phase) * 0.9)));
      });

  return (
    <div className={`waveform ${active ? "live" : ""}`} aria-label={active ? "Live recording waveform" : "Recorded audio waveform"}>
      <div className="waveform-ruler">
        <span>0:00</span>
        <span>timeline</span>
        <span>clip end</span>
      </div>
      <div className="waveform-bars">
        {displayBars.map((bar, index) => (
          <span key={index} style={{ height: `${Math.max(8, bar * 92)}%` }} />
        ))}
      </div>
    </div>
  );
}
