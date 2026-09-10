import { useState } from "react";
import { Panel } from "../../components/ui";
import { formatDuration } from "../../constants/asr";
import { useRecorder } from "../../hooks/useRecorder";
import { useTranscriptionOptions } from "../../hooks/useTranscriptionOptions";
import { postForm } from "../../lib/api";
import type { LiveDebugState, OutputView, TranscriptionPayload } from "../../types/asr";
import { AudioSourcePicker } from "./components/AudioSourcePicker";
import { BackendOutputPanel } from "./components/BackendOutputPanel";
import { DecodeOptionsPanel } from "./components/DecodeOptionsPanel";
import { MicBadge } from "./components/MicBadge";
import { RecordingConsole } from "./components/RecordingConsole";
import { SourceSummary } from "./components/SourceSummary";

export function LiveMicAudio() {
  const recorder = useRecorder();
  const { options, setters, appendOptions } = useTranscriptionOptions();
  const [outputView, setOutputView] = useState<OutputView>("transcript");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState<TranscriptionPayload | null>(null);
  const [submittedAt, setSubmittedAt] = useState<Date | null>(null);
  const [completedAt, setCompletedAt] = useState<Date | null>(null);
  const activeDuration = recorder.micState === "recording" ? recorder.recordingSeconds : recorder.recordedDuration;
  const debug: LiveDebugState = {
    status: busy ? "transcribing" : error ? "failed" : payload ? "complete" : recorder.micState === "recording" ? "recording" : recorder.sourceReady ? "ready" : "idle",
    source: {
      type: recorder.file ? "upload" : recorder.recorded ? "microphone" : "none",
      name: recorder.file?.name ?? (recorder.recorded ? "recording.webm" : ""),
      mime_type: recorder.file?.type ?? recorder.previewMimeType,
      duration_sec: Number(activeDuration.toFixed(2)),
    },
    request: {
      model_name: options.model,
      language: options.language,
      beam_size: options.beam,
      temperature: options.temperature,
      initial_prompt: options.prompt,
      vad_filter: options.vad,
      profile: options.profile,
      chunk_length: options.chunkLength,
      vad_aggressiveness: options.vadAggressiveness,
      condition_on_previous_text: options.conditionOnPreviousText,
      repetition_guard: options.repetitionGuard,
      hotwords: options.hotwords,
      output_script: options.outputScript,
    },
    timing: {
      submitted_at: submittedAt?.toISOString(),
      completed_at: completedAt?.toISOString(),
      elapsed_ms: submittedAt && completedAt ? completedAt.getTime() - submittedAt.getTime() : undefined,
    },
    error: error || undefined,
    backend_quality: payload?.result?.quality,
  };

  function resetLiveState() {
    recorder.resetSource();
    setPayload(null);
    setError("");
    setSubmittedAt(null);
    setCompletedAt(null);
    setOutputView("transcript");
  }

  async function handleFile(file: File | null) {
    setPayload(null);
    setCompletedAt(null);
    setOutputView("transcript");
    await recorder.handleFile(file);
  }

  async function startRecording() {
    setError("");
    setPayload(null);
    setCompletedAt(null);
    await recorder.startRecording();
  }

  async function submit() {
    const source = recorder.file ?? (recorder.recorded ? new File([recorder.recorded], "recording.webm", { type: "audio/webm" }) : null);
    if (!source) return setError("Record speech or choose an audio file first.");
    const started = new Date();
    setSubmittedAt(started);
    setCompletedAt(null);
    setBusy(true);
    setError("");
    setOutputView("transcript");
    try {
      const form = new FormData();
      form.append("audio", source);
      appendOptions(form);
      setPayload(await postForm("/api/transcribe", form));
      setCompletedAt(new Date());
    } catch (exc: any) {
      setError(exc.message);
      setCompletedAt(new Date());
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="workspace live-grid">
      <Panel title="Live Mic & Audio" action={<MicBadge state={recorder.micState} />}>
        <RecordingConsole
          micState={recorder.micState}
          sourceReady={recorder.sourceReady}
          duration={activeDuration}
          inputLevel={recorder.inputLevel}
          waveform={recorder.waveform}
          previewUrl={recorder.previewUrl}
          previewMimeType={recorder.previewMimeType}
          busy={busy}
          onStart={startRecording}
          onStop={recorder.stopRecording}
          onCheckMic={recorder.checkMic}
          onTranscribe={submit}
          onClear={resetLiveState}
        />
        <div className="source-panel">
          <AudioSourcePicker file={recorder.file} onChange={handleFile} />
          <SourceSummary file={recorder.file} recorded={recorder.recorded} duration={activeDuration} sourceReady={recorder.sourceReady} />
        </div>
        <DecodeOptionsPanel options={options} setters={setters} />
        {error && <p className="error">{error}</p>}
        {recorder.sourceReady && <p className="source-note">Current clip length: {formatDuration(activeDuration)}</p>}
      </Panel>
      <BackendOutputPanel payload={payload} busy={busy} debug={debug} outputView={outputView} onOutputViewChange={setOutputView} />
    </div>
  );
}
