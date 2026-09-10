import { Waves } from "lucide-react";
import { EmptyState, ExportLinks, Panel } from "../../../components/ui";
import type { LiveDebugState, OutputView, TranscriptionPayload } from "../../../types/asr";
import { DebugView } from "./DebugView";
import { OutputTabs } from "./OutputTabs";
import { RawJsonView } from "./RawJsonView";
import { SegmentsTable } from "./SegmentsTable";
import { TranscribingState } from "./TranscribingState";
import { TranscriptView } from "./TranscriptView";

export function BackendOutputPanel({
  payload,
  busy,
  debug,
  outputView,
  onOutputViewChange
}: {
  payload: TranscriptionPayload | null;
  busy: boolean;
  debug: LiveDebugState;
  outputView: OutputView;
  onOutputViewChange: (view: OutputView) => void;
}) {
  const result = payload?.result;
  return (
    <Panel title="Backend Output" action={result && <ExportLinks exports={payload.exports} />}>
      <OutputTabs value={outputView} onChange={onOutputViewChange} />
      {busy && outputView !== "debug" && <TranscribingState />}
      {!busy && !result && outputView !== "debug" && (
        <EmptyState className="structured-empty">
          <Waves size={32} />
          <strong>No backend result yet</strong>
          <span>Record or upload audio, then transcribe to populate transcript, time segments, metadata, and export links.</span>
        </EmptyState>
      )}
      {!busy && result && outputView === "transcript" && <TranscriptView result={result} />}
      {!busy && result && outputView === "segments" && <SegmentsTable rows={result.segments ?? []} />}
      {!busy && result && outputView === "raw" && <RawJsonView payload={payload} />}
      {outputView === "debug" && <DebugView debug={debug} payload={payload} />}
    </Panel>
  );
}
