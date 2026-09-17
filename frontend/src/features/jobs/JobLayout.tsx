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
            {job.kind === "train" && job.result?.analysis && <TrainingAnalysis result={job.result} />}
            {!logsOnly && rows && <ResultTable rows={rows} />}
          </>
        ) : (
          <EmptyState>Start a job to see progress, logs, and downloads.</EmptyState>
        )}
      </Panel>
    </div>
  );
}

function pct(value: unknown) {
  return `${((Number(value) || 0) * 100).toFixed(2)}%`;
}

function TrainingAnalysis({ result }: { result: any }) {
  const bars = result.analysis?.charts?.wer_cer_bar ?? [];
  const oldMatrix = result.analysis?.old_confusion_matrix;
  const newMatrix = result.analysis?.new_confusion_matrix;
  const maxMetric = Math.max(0.01, ...bars.flatMap((row: any) => [Number(row.wer) || 0, Number(row.cer) || 0]));
  return (
    <div className="analysis-panel">
      <h3>Model decision analytics</h3>
      <div className="analysis-cards">
        <span>Verdict <strong>{result.verdict}</strong></span>
        <span>Old WER <strong>{pct(result.old?.corpus_wer ?? result.old?.mean_sample_wer)}</strong></span>
        <span>New WER <strong>{pct(result.new?.corpus_wer ?? result.new?.mean_sample_wer)}</strong></span>
        <span>Old CER <strong>{pct(result.old?.corpus_cer ?? result.old?.mean_sample_cer)}</strong></span>
        <span>New CER <strong>{pct(result.new?.corpus_cer ?? result.new?.mean_sample_cer)}</strong></span>
      </div>
      <div className="chart-card">
        <h4>WER / CER comparison</h4>
        <svg className="bar-chart" viewBox="0 0 420 170" role="img" aria-label="Old and new model WER CER bar chart">
          {bars.map((row: any, index: number) => {
            const x = 70 + index * 170;
            const werHeight = (Number(row.wer) / maxMetric) * 110;
            const cerHeight = (Number(row.cer) / maxMetric) * 110;
            return (
              <g key={row.model}>
                <rect x={x} y={135 - werHeight} width="42" height={werHeight} fill="#087a5d" />
                <rect x={x + 52} y={135 - cerHeight} width="42" height={cerHeight} fill="#2563eb" />
                <text x={x + 47} y="158" textAnchor="middle">{row.model}</text>
              </g>
            );
          })}
          <text x="12" y="24">Lower is better</text>
          <text x="300" y="24" fill="#087a5d">WER</text>
          <text x="350" y="24" fill="#2563eb">CER</text>
        </svg>
      </div>
      <div className="matrix-grid">
        <ConfusionSummary title="Previous model confusion" matrix={oldMatrix} />
        <ConfusionSummary title="New model confusion" matrix={newMatrix} />
      </div>
    </div>
  );
}

function ConfusionSummary({ title, matrix }: { title: string; matrix: any }) {
  if (!matrix) return null;
  const rows = matrix.top_substitutions?.slice(0, 8) ?? [];
  return (
    <div className="chart-card">
      <h4>{title}</h4>
      <div className="mini-metrics">
        <span>Exact: {matrix.token_totals?.exact ?? 0}</span>
        <span>Sub: {matrix.token_totals?.substitutions ?? 0}</span>
        <span>Del: {matrix.token_totals?.deletions ?? 0}</span>
        <span>Ins: {matrix.token_totals?.insertions ?? 0}</span>
      </div>
      <table className="compact-table">
        <thead><tr><th>Expected</th><th>Predicted</th><th>Count</th></tr></thead>
        <tbody>{rows.map((row: any) => <tr key={`${row.expected}-${row.predicted}`}><td>{row.expected}</td><td>{row.predicted}</td><td>{row.count}</td></tr>)}</tbody>
      </table>
    </div>
  );
}

function ResultTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  const columns = useMemo(() => Object.keys(rows[0] ?? {}), [rows]);
  return <DataTable columns={columns} rows={rows} />;
}
