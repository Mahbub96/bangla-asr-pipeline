import { Spinner } from "../../../components/ui";

export function TranscribingState() {
  return (
    <div className="structured-empty transcribing-state">
      <Spinner label="Transcribing audio" />
      <strong>Transcribing audio</strong>
      <span>The backend is decoding the uploaded or recorded clip. Longer Bangla audio can take a little while on CPU.</span>
    </div>
  );
}
