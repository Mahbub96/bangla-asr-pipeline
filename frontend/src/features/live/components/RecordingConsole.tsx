import { Mic, Play, RefreshCw, Square, Trash2 } from "lucide-react";
import type { MicState } from "../../../lib/media";
import { Button, Spinner } from "../../../components/ui";
import { InputLevelMeter } from "./InputLevelMeter";
import { RecordingStatus } from "./RecordingStatus";
import { RecordingTimer } from "./RecordingTimer";
import { WaveformViewer } from "./WaveformViewer";
import { AudioPreview } from "./AudioPreview";

export function RecordingConsole({
  micState,
  sourceReady,
  duration,
  inputLevel,
  waveform,
  previewUrl,
  previewMimeType,
  busy,
  onStart,
  onStop,
  onCheckMic,
  onTranscribe,
  onClear
}: {
  micState: MicState;
  sourceReady: boolean;
  duration: number;
  inputLevel: number;
  waveform: number[];
  previewUrl: string;
  previewMimeType: string;
  busy: boolean;
  onStart: () => void;
  onStop: () => void;
  onCheckMic: () => void;
  onTranscribe: () => void;
  onClear: () => void;
}) {
  const recording = micState === "recording";
  return (
    <div className={`recording-console ${recording ? "is-recording" : ""}`}>
      <div className="recording-primary">
        <RecordingStatus state={micState} sourceReady={sourceReady} />
        <RecordingTimer seconds={duration} />
      </div>
      <InputLevelMeter level={inputLevel} />
      <WaveformViewer bars={waveform} active={recording} level={inputLevel} />
      <AudioPreview url={previewUrl} mimeType={previewMimeType} />
      <div className="recording-actions">
        <Button onClick={recording ? onStop : onStart} className={recording ? "danger" : "primary"}>
          {recording ? <Square size={16} /> : <Mic size={16} />}
          {recording ? "Stop Recording" : "Start Recording"}
        </Button>
        <Button onClick={onCheckMic} disabled={recording}>
          <RefreshCw size={16} />
          Check Mic
        </Button>
        <Button className="primary" onClick={onTranscribe} disabled={busy || !sourceReady || recording}>
          {busy ? <Spinner label="Transcribing audio" /> : <Play size={16} />}
          {busy ? "Transcribing" : "Transcribe Audio"}
        </Button>
        <Button onClick={onClear} disabled={recording}>
          <Trash2 size={16} />
          Clear
        </Button>
      </div>
    </div>
  );
}
