import { Activity } from "lucide-react";
import { useState } from "react";
import { Button, Field } from "../../components/ui";
import { ASR_LANGUAGES, ASR_MODELS, DEFAULT_AUDIO_DIRECTORY, DEFAULT_METADATA_CSV } from "../../constants/asr";
import { useJobStream } from "../../hooks/useJobStream";
import { postForm } from "../../lib/api";
import { JobLayout } from "../jobs/JobLayout";

export function Benchmark() {
  const [csv, setCsv] = useState<File | null>(null);
  const [csvPath, setCsvPath] = useState(DEFAULT_METADATA_CSV);
  const [audioDir, setAudioDir] = useState(DEFAULT_AUDIO_DIRECTORY);
  const [model, setModel] = useState("large-v3-turbo");
  const [language, setLanguage] = useState("auto");
  const { job, error, setError, attach, cancel } = useJobStream();

  async function submit() {
    try {
      const form = new FormData();
      if (csv) form.append("csv_file", csv);
      form.append("metadata_csv_path", csvPath);
      form.append("audio_dir", audioDir);
      form.append("model_name", model);
      form.append("language", language);
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
        <Field label="Model">
          <select value={model} onChange={(event) => setModel(event.target.value)}>
            {ASR_MODELS.map((item) => <option key={item}>{item}</option>)}
          </select>
        </Field>
        <Field label="Language">
          <select value={language} onChange={(event) => setLanguage(event.target.value)}>
            {ASR_LANGUAGES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
        </Field>
      </div>
      <Button className="primary" onClick={submit}>
        <Activity size={16} />
        Run Benchmark
      </Button>
    </JobLayout>
  );
}
