import type { MicState } from "../../../lib/media";

export function RecordingStatus({ state, sourceReady }: { state: MicState; sourceReady: boolean }) {
  return (
    <div className="recording-state">
      <span className="recording-dot" />
      <span>{state === "recording" ? "Recording in progress" : sourceReady ? "Audio source ready" : "Waiting for audio"}</span>
    </div>
  );
}
