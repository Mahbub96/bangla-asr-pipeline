import { Braces, Bug, FileAudio, ListTree } from "lucide-react";
import type { OutputView } from "../../../types/asr";

export function OutputTabs({ value, onChange }: { value: OutputView; onChange: (view: OutputView) => void }) {
  return (
    <div className="output-toolbar" role="tablist" aria-label="Transcription output views">
      <button className={value === "transcript" ? "active" : ""} onClick={() => onChange("transcript")}>
        <FileAudio size={16} />
        Raw transcript
      </button>
      <button className={value === "segments" ? "active" : ""} onClick={() => onChange("segments")}>
        <ListTree size={16} />
        Time segments
      </button>
      <button className={value === "raw" ? "active" : ""} onClick={() => onChange("raw")}>
        <Braces size={16} />
        API JSON
      </button>
      <button className={value === "debug" ? "active" : ""} onClick={() => onChange("debug")}>
        <Bug size={16} />
        Debug
      </button>
    </div>
  );
}
