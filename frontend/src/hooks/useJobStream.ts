import { useState } from "react";
import { postJson } from "../lib/api";
import type { JobResponse, JobSnapshot } from "../types/asr";

export function useJobStream() {
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [error, setError] = useState("");

  function attach(response: JobResponse) {
    setError("");
    const events = new EventSource(response.events_url);
    events.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setJob(data.job);
      if (["done"].includes(data.type)) events.close();
    };
    events.onerror = () => {
      events.close();
      setError("Progress stream disconnected. Refresh job status if needed.");
    };
  }

  async function cancel() {
    if (!job) return;
    setJob(await postJson(`/api/jobs/${job.id}/cancel`, {}));
  }

  return { job, setJob, error, setError, attach, cancel };
}
