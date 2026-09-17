import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { Button, ConsoleLog, EmptyState, Panel } from "../../components/ui";
import { getJson } from "../../lib/api";
import type { JobSnapshot, RuntimeLog } from "../../types/asr";

type RuntimeLogsPayload = { logs: RuntimeLog[] };
type ActiveJobsPayload = { jobs: JobSnapshot[] };

export function Logs() {
  const [runtimeLogs, setRuntimeLogs] = useState<RuntimeLog[]>([]);
  const [activeJobs, setActiveJobs] = useState<JobSnapshot[]>([]);
  const [error, setError] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);

  async function refresh() {
    try {
      const [runtime, active] = await Promise.all([
        getJson<RuntimeLogsPayload>("/api/logs/runtime?limit=500"),
        getJson<ActiveJobsPayload>("/api/logs/jobs/active"),
      ]);
      setRuntimeLogs(runtime.logs);
      setActiveJobs(active.jobs);
      setError("");
    } catch (exc: any) {
      setError(exc.message);
    }
  }

  useEffect(() => {
    refresh();
    if (!autoRefresh) return;
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, [autoRefresh]);

  const runtimeLines = runtimeLogs.map((item) => item.message);
  const jobLines = activeJobs.flatMap((job) => [
    `# ${job.kind} ${job.id} — ${job.status} — ${job.message}`,
    ...job.logs,
  ]);

  return (
    <div className="workspace two-col">
      <Panel
        title="Live Runtime Logs"
        action={
          <div className="log-actions">
            <label><input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} /> Auto refresh</label>
            <Button onClick={refresh}><RefreshCw size={16} /> Refresh</Button>
          </div>
        }
      >
        {error && <p className="error">{error}</p>}
        {runtimeLines.length ? <ConsoleLog logs={runtimeLines} /> : <EmptyState>No backend runtime logs yet.</EmptyState>}
      </Panel>
      <Panel title="Active Job Logs">
        {jobLines.length ? <ConsoleLog logs={jobLines} /> : <EmptyState>No active jobs. Training logs appear here while a training job is running.</EmptyState>}
      </Panel>
    </div>
  );
}
