import { useState } from "react";

export function useWaveform() {
  const [waveform, setWaveform] = useState<number[]>([]);
  const [duration, setDuration] = useState(0);

  async function buildWaveform(blob: Blob) {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const context = new AudioCtx();
    try {
      const buffer = await context.decodeAudioData(await blob.arrayBuffer());
      const channel = buffer.getChannelData(0);
      const bars = 120;
      const block = Math.max(1, Math.floor(channel.length / bars));
      const nextWaveform = Array.from({ length: bars }, (_, index) => {
        let peak = 0;
        const start = index * block;
        for (let i = start; i < Math.min(start + block, channel.length); i += 1) {
          peak = Math.max(peak, Math.abs(channel[i]));
        }
        return Math.max(0.04, Math.min(1, peak));
      });
      setWaveform(nextWaveform);
      setDuration(buffer.duration);
    } finally {
      await context.close();
    }
  }

  function resetWaveform() {
    setWaveform([]);
    setDuration(0);
  }

  return { waveform, duration, setDuration, setWaveform, buildWaveform, resetWaveform };
}
