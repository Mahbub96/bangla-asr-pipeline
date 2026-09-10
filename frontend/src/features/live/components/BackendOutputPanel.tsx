import { Waves } from "lucide-react";
import { EmptyState, ExportLinks, Panel } from "../../../components/ui";
import type { OutputView, TranscriptionPayload } from "../../../types/asr";
import { OutputTabs } from "./OutputTabs";
import { RawJsonView } from "./RawJsonView";
import { SegmentsTable } from "./SegmentsTable";
import { TranscriptView } from "./TranscriptView";

export function BackendOutputPanel({
  payload,
  outputView,
  onOutputViewChange
}: {
  payload: TranscriptionPayload | null;
  outputView: OutputView;
  onOutputViewChange: (view: OutputView) => void;
}) {
  const result = payload?.result;
  return (
    <Panel title="Backend Output" action={result && <ExportLinks exports={payload.exports} />}>
      <OutputTabs value={outputView} onChange={onOutputViewChange} />
      {!result && (
        <EmptyState className="structured-empty">
          <Waves size={32} />
          <strong>No backend result yet</strong>
          <span>Record or upload audio, then transcribe to populate transcript, time segments, metadata, and export links.</span>
        </EmptyState>
      )}
      {result && outputView === "transcript" && <TranscriptView result={result} />}
      {result && outputView === "segments" && <SegmentsTable rows={result.segments ?? []} />}
      {result && outputView === "raw" && <RawJsonView payload={payload} />}
    </Panel>
  );
}
