import { Clock } from "lucide-react";
import { formatDuration } from "../../../constants/asr";

export function RecordingTimer({ seconds }: { seconds: number }) {
  return (
    <div className="recording-time">
      <Clock size={18} />
      {formatDuration(seconds)}
    </div>
  );
}
