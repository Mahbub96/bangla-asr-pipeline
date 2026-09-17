import { Terminal } from "lucide-react";
import { useEffect, useState } from "react";
import { Button, Field } from "../../components/ui";
import { DEFAULT_TRAINING_CONFIG } from "../../constants/asr";
import { useJobStream } from "../../hooks/useJobStream";
import { postJson } from "../../lib/api";
import type { TrainingConfig } from "../../types/asr";
import { JobLayout } from "../jobs/JobLayout";

const ACTIVE_TRAINING_JOB_KEY = "bangla-asr-active-training-job-id";
const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function Training() {
  const { job, setJob, error, setError, attach, cancel } = useJobStream();
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);

  useEffect(() => {
    const jobId = localStorage.getItem(ACTIVE_TRAINING_JOB_KEY);
    if (!jobId) return;

    let cancelled = false;
    fetch(`/api/jobs/${jobId}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((snapshot) => {
        if (cancelled) return;
        if (!snapshot) {
          localStorage.removeItem(ACTIVE_TRAINING_JOB_KEY);
          return;
        }
        setJob(snapshot);
        if (!TERMINAL_STATUSES.has(snapshot.status)) {
          attach({ job_id: jobId, status_url: `/api/jobs/${jobId}`, events_url: `/api/jobs/${jobId}/events` });
        } else {
          localStorage.removeItem(ACTIVE_TRAINING_JOB_KEY);
        }
      })
      .catch(() => setError("Could not restore the active training job after reload."));

    return () => {
      cancelled = true;
    };
  }, [attach, setError, setJob]);

  useEffect(() => {
    if (job && TERMINAL_STATUSES.has(job.status)) {
      localStorage.removeItem(ACTIVE_TRAINING_JOB_KEY);
    }
  }, [job]);

  function set(key: keyof TrainingConfig, value: string | number | boolean | null) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  async function submit() {
    try {
      const response = await postJson<{ job_id: string; status_url: string; events_url: string }>("/api/train/jobs", config);
      localStorage.setItem(ACTIVE_TRAINING_JOB_KEY, response.job_id);
      attach(response);
    } catch (exc: any) {
      setError(exc.message);
    }
  }

  return (
    <JobLayout title="Batched Training" job={job} error={error} onCancel={cancel} logsOnly>
      <div className="notice-card">
        Recommended Mac mini M4 smoke-test values are prefilled. You can click Launch Training without changing anything.
      </div>
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
        Launch Training with Recommended Values
      </Button>
    </JobLayout>
  );
}
