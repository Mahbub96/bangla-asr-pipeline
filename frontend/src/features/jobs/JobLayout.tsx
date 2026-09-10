import { Square } from "lucide-react";
import { useMemo, type ReactNode } from "react";
import { Button, ConsoleLog, DataTable, EmptyState, ExportLinks, Panel, StatusPill } from "../../components/ui";
import type { JobSnapshot } from "../../types/asr";

export function JobLayout({
  title,
  children,
  job,
  error,
  onCancel,
  logsOnly = false
}: {
  title: string;
  children: ReactNode;
  job: JobSnapshot | null;
  error: string;
  onCancel: () => void;
  logsOnly?: boolean;
}) {
  const rows = Array.isArray(job?.result) ? job?.result : job?.result?.rows;
  return (
    <div className="workspace two-col">
      <Panel title={title}>
        {children}
        {error && <p className="error">{error}</p>}
      </Panel>
      <Panel title="Progress" action={job && <Button onClick={onCancel} disabled={!["queued", "running"].includes(job.status)}><Square size={16} /> Cancel</Button>}>
        {job ? (
          <>
            <div className="job-head">
              <StatusPill tone={job.status === "completed" ? "good" : job.status === "failed" ? "bad" : "live"}>{job.status}</StatusPill>
              <span>{job.message}</span>
            </div>
            <progress value={job.progress} max={1} />
            <ConsoleLog logs={job.logs} />
            <ExportLinks exports={job.exports} />
            {!logsOnly && rows && <ResultTable rows={rows} />}
          </>
        ) : (
          <EmptyState>Start a job to see progress, logs, and downloads.</EmptyState>
        )}
      </Panel>
    </div>
  );
}

function ResultTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  const columns = useMemo(() => Object.keys(rows[0] ?? {}), [rows]);
  return <DataTable columns={columns} rows={rows} />;
}
