import { StatusPill } from "../../../components/ui";
import type { MicState } from "../../../lib/media";

export function MicBadge({ state }: { state: MicState }) {
  const tone = state === "ready" || state === "stopped" ? "good" : state === "recording" ? "live" : state === "checking" ? "neutral" : "bad";
  const labels: Record<MicState, string> = {
    checking: "Checking mic",
    unsupported: "Mic unsupported",
    insecure: "Use localhost or HTTPS",
    "permission-denied": "Permission blocked",
    "no-device": "No mic detected",
    ready: "Mic ready",
    recording: "Recording",
    stopped: "Recorded",
    failed: "Mic unavailable"
  };
  return <StatusPill tone={tone}>{labels[state]}</StatusPill>;
}
