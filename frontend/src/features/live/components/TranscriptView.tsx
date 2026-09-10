import { MetricGrid } from "../../../components/ui";
import type { TranscriptionResult } from "../../../types/asr";

export function TranscriptView({ result }: { result: TranscriptionResult }) {
  const quality = result.quality;
  return (
    <>
      {quality && (
        <div className="quality-strip">
          <span className={quality.low_confidence ? "quality-badge bad" : "quality-badge good"}>Confidence {Math.round(quality.language_probability * 100)}%</span>
          <span className="quality-badge">Profile {quality.profile}</span>
          <span className={quality.repetition_score > 0.12 ? "quality-badge warn" : "quality-badge good"}>Repetition {Math.round(quality.repetition_score * 100)}%</span>
        </div>
      )}
      {quality?.warnings?.length ? (
        <div className="quality-warnings">
          {quality.warnings.map((warning) => <span key={warning}>{warning}</span>)}
          {quality.suggested_retry && <strong>{quality.suggested_retry}</strong>}
        </div>
      ) : null}
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
