import { Activity } from "lucide-react";
import { useState } from "react";
import { Button, Field } from "../../components/ui";
import { DEFAULT_AUDIO_DIRECTORY, DEFAULT_METADATA_CSV } from "../../constants/asr";
import { useJobStream } from "../../hooks/useJobStream";
import { useTranscriptionOptions } from "../../hooks/useTranscriptionOptions";
import { postForm } from "../../lib/api";
import { DecodeOptionsPanel } from "../live/components/DecodeOptionsPanel";
import { JobLayout } from "../jobs/JobLayout";

export function Benchmark() {
  const [csv, setCsv] = useState<File | null>(null);
  const [csvPath, setCsvPath] = useState(DEFAULT_METADATA_CSV);
  const [audioDir, setAudioDir] = useState(DEFAULT_AUDIO_DIRECTORY);
  const { options, setters, appendOptions } = useTranscriptionOptions();
  const [compareProfiles, setCompareProfiles] = useState(false);
  const { job, error, setError, attach, cancel } = useJobStream();

  async function submit() {
    try {
      const form = new FormData();
      if (csv) form.append("csv_file", csv);
      form.append("metadata_csv_path", csvPath);
      form.append("audio_dir", audioDir);
      appendOptions(form);
      form.append("compare_profiles", String(compareProfiles));
      attach(await postForm("/api/evaluate/jobs", form));
    } catch (exc: any) {
      setError(exc.message);
    }
  }

  return (
    <JobLayout title="Benchmark WER / CER" job={job} error={error} onCancel={cancel}>
      <input type="file" accept=".csv" onChange={(event) => setCsv(event.target.files?.[0] ?? null)} />
      <div className="option-grid">
        <Field label="Metadata CSV">
          <input value={csvPath} onChange={(event) => setCsvPath(event.target.value)} />
        </Field>
        <Field label="Audio directory">
          <input value={audioDir} onChange={(event) => setAudioDir(event.target.value)} />
        </Field>
      </div>
      <DecodeOptionsPanel options={options} setters={setters} />
      <label className="check">
        <input type="checkbox" checked={compareProfiles} onChange={(event) => setCompareProfiles(event.target.checked)} />
        Compare accuracy profiles
      </label>
      <Button className="primary" onClick={submit}>
        <Activity size={16} />
        Run Benchmark
      </Button>
    </JobLayout>
  );
}
