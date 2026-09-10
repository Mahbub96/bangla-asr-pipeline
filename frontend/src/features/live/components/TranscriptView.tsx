import { MetricGrid } from "../../../components/ui";
import type { TranscriptionResult } from "../../../types/asr";

export function TranscriptView({ result }: { result: TranscriptionResult }) {
  return (
    <>
      <textarea className="transcript enhanced" value={result.text ?? ""} readOnly />
      <MetricGrid
        items={[
          { label: "Language", value: result.language },
          { label: "Confidence", value: `${Math.round(result.language_probability * 100)}%` },
          { label: "Duration", value: `${result.duration_sec}s` },
          { label: "Speed", value: `${result.speed_factor}x` }
        ]}
      />
    </>
  );
}
