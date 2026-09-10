import { formatDuration } from "../../../constants/asr";
import { Info } from "../../../components/ui";

export function SourceSummary({
  file,
  recorded,
  duration,
  sourceReady
}: {
  file: File | null;
  recorded: Blob | null;
  duration: number;
  sourceReady: boolean;
}) {
  return (
    <div className="source-summary">
      <Info label="Source" value={file ? "Uploaded file" : recorded ? "Microphone recording" : "No audio selected"} />
      <Info label="Duration" value={sourceReady ? formatDuration(duration) : "0:00.0"} />
    </div>
  );
}
