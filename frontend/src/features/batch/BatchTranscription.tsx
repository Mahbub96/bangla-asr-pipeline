import { FileAudio } from "lucide-react";
import { useState } from "react";
import { Button, Field, SegmentedControl } from "../../components/ui";
import { DEFAULT_BATCH_DIRECTORY } from "../../constants/asr";
import { useJobStream } from "../../hooks/useJobStream";
import { useTranscriptionOptions } from "../../hooks/useTranscriptionOptions";
import { postForm } from "../../lib/api";
import { DecodeOptionsPanel } from "../live/components/DecodeOptionsPanel";
import { JobLayout } from "../jobs/JobLayout";

type BatchMode = "files" | "directory";

export function BatchTranscription() {
  const { options, setters, appendOptions } = useTranscriptionOptions();
  const [files, setFiles] = useState<FileList | null>(null);
  const [directory, setDirectory] = useState(DEFAULT_BATCH_DIRECTORY);
  const [mode, setMode] = useState<BatchMode>("files");
  const { job, error, setError, attach, cancel } = useJobStream();

  async function submit() {
    try {
      const form = new FormData();
      appendOptions(form);
      if (mode === "files" && files) Array.from(files).forEach((file) => form.append("files", file));
      if (mode === "directory") form.append("directory_path", directory);
      attach(await postForm("/api/batch/jobs", form));
    } catch (exc: any) {
      setError(exc.message);
    }
  }

  return (
    <JobLayout title="Batch Transcription" job={job} error={error} onCancel={cancel}>
      <SegmentedControl
        value={mode}
        options={[
          { label: "Upload Files", value: "files" },
          { label: "Server Directory", value: "directory" }
        ]}
        onChange={setMode}
      />
      {mode === "files" ? (
        <input type="file" multiple accept="audio/*" onChange={(event) => setFiles(event.target.files)} />
      ) : (
        <Field label="Directory path">
          <input value={directory} onChange={(event) => setDirectory(event.target.value)} />
        </Field>
      )}
      <DecodeOptionsPanel options={options} setters={setters} />
      <Button className="primary" onClick={submit}>
        <FileAudio size={16} />
        Start Batch
      </Button>
    </JobLayout>
  );
}
