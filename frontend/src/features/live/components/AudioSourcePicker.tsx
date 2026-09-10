import { FileDropzone } from "../../../components/ui";

export function AudioSourcePicker({ file, onChange }: { file: File | null; onChange: (file: File | null) => void }) {
  return <FileDropzone accept="audio/*,video/*" label={file ? file.name : "Choose an existing audio or video file"} onChange={(files) => onChange(files?.[0] ?? null)} />;
}
