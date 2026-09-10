import { useEffect, useRef, useState } from "react";
import { checkMicrophone, type MicState } from "../lib/media";
import { useAudioMeter } from "./useAudioMeter";
import { useWaveform } from "./useWaveform";

export function useRecorder() {
  const [micState, setMicState] = useState<MicState>("checking");
  const [file, setFile] = useState<File | null>(null);
  const [recorded, setRecorded] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewMimeType, setPreviewMimeType] = useState("");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const recordingStartedAt = useRef(0);
  const { waveform, duration, setDuration, setWaveform, buildWaveform, resetWaveform } = useWaveform();
  const { inputLevel, startMeter, stopMeter } = useAudioMeter(() => {
    setRecordingSeconds((performance.now() - recordingStartedAt.current) / 1000);
  });

  useEffect(() => {
    checkMicrophone(false).then(setMicState);
    return () => stopMeter();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function clearPreviewUrl() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl("");
    setPreviewMimeType("");
  }

  function setPreviewSource(source: Blob | File) {
    clearPreviewUrl();
    setPreviewUrl(URL.createObjectURL(source));
    setPreviewMimeType(source.type);
  }

  function setRecordingBlob(blob: Blob) {
    setPreviewSource(blob);
    setRecorded(blob);
    buildWaveform(blob).catch(() => setWaveform([]));
  }

  function resetSource() {
    setFile(null);
    setRecorded(null);
    setRecordingSeconds(0);
    resetWaveform();
    clearPreviewUrl();
  }

  async function handleFile(fileValue: File | null) {
    setFile(fileValue);
    clearPreviewUrl();
    if (!fileValue) return;
    setRecorded(null);
    resetWaveform();
    setPreviewSource(fileValue);
    if (fileValue.type.startsWith("audio/")) {
      await buildWaveform(fileValue).catch(() => setWaveform([]));
    }
  }

  async function startRecording() {
    setFile(null);
    setRecorded(null);
    setDuration(0);
    setWaveform([]);
    clearPreviewUrl();
    const state = await checkMicrophone(true);
    if (state !== "ready") {
      setMicState(state);
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks.current = [];
    recordingStartedAt.current = performance.now();
    setRecordingSeconds(0);
    recorder.current = new MediaRecorder(stream);
    recorder.current.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.current.push(event.data);
    };
    recorder.current.onstop = () => {
      const clipDuration = (performance.now() - recordingStartedAt.current) / 1000;
      const blob = new Blob(chunks.current, { type: "audio/webm" });
      stopMeter();
      setDuration(clipDuration);
      setRecordingBlob(blob);
      setMicState("stopped");
    };
    recorder.current.start();
    startMeter(stream);
    setMicState("recording");
  }

  function stopRecording() {
    recorder.current?.stop();
  }

  function checkMic() {
    return checkMicrophone(true).then(setMicState);
  }

  return {
    micState,
    setMicState,
    file,
    recorded,
    previewUrl,
    previewMimeType,
    recordedDuration: duration,
    recordingSeconds,
    inputLevel,
    waveform,
    sourceReady: Boolean(recorded || file),
    handleFile,
    startRecording,
    stopRecording,
    resetSource,
    checkMic
  };
}
