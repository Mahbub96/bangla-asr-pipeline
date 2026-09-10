import { useRef, useState } from "react";

export function useAudioMeter(onTick?: () => void) {
  const [inputLevel, setInputLevel] = useState(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  function stopMeter() {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setInputLevel(0);
  }

  function startMeter(stream: MediaStream) {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const context = new AudioCtx();
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    context.createMediaStreamSource(stream).connect(analyser);
    audioContextRef.current = context;
    streamRef.current = stream;
    const data = new Uint8Array(analyser.fftSize);

    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (const sample of data) {
        const normalized = (sample - 128) / 128;
        sum += normalized * normalized;
      }
      const rms = Math.sqrt(sum / data.length);
      setInputLevel(Math.min(1, rms * 5));
      onTick?.();
      animationRef.current = requestAnimationFrame(tick);
    };
    tick();
  }

  return { inputLevel, startMeter, stopMeter };
}
