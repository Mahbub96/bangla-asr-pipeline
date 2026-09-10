import type { LiveDebugState, TranscriptionPayload } from "../../../types/asr";

export function DebugView({ debug, payload }: { debug: LiveDebugState; payload: TranscriptionPayload | null }) {
  return (
    <pre className="json-view debug-view">
      {JSON.stringify({ debug, response: payload }, null, 2)}
    </pre>
  );
}
