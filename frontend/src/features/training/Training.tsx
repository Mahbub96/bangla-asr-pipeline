import { Terminal } from "lucide-react";
import { useState } from "react";
import { Button, Field } from "../../components/ui";
import { DEFAULT_TRAINING_CONFIG } from "../../constants/asr";
import { useJobStream } from "../../hooks/useJobStream";
import { postJson } from "../../lib/api";
import type { TrainingConfig } from "../../types/asr";
import { JobLayout } from "../jobs/JobLayout";

export function Training() {
  const { job, error, setError, attach, cancel } = useJobStream();
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);

  function set(key: keyof TrainingConfig, value: string | number | boolean | null) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  async function submit() {
    try {
      attach(await postJson("/api/train/jobs", config));
    } catch (exc: any) {
      setError(exc.message);
    }
  }

  return (
    <JobLayout title="Batched Training" job={job} error={error} onCancel={cancel} logsOnly>
      <div className="option-grid">
        {Object.entries(config).map(([key, value]) => (
          <Field key={key} label={key.replaceAll("_", " ")}>
            {typeof value === "boolean" ? (
              <input type="checkbox" checked={value} onChange={(event) => set(key as keyof TrainingConfig, event.target.checked)} />
            ) : value === null ? (
              <input placeholder="optional" onChange={(event) => set(key as keyof TrainingConfig, event.target.value ? Number(event.target.value) : null)} />
            ) : (
              <input value={value} onChange={(event) => set(key as keyof TrainingConfig, typeof value === "number" ? Number(event.target.value) : event.target.value)} />
            )}
          </Field>
        ))}
      </div>
      <Button className="primary" onClick={submit}>
        <Terminal size={16} />
        Launch Training
      </Button>
    </JobLayout>
  );
}
