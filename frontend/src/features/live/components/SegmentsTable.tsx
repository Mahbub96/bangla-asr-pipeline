import { EmptyState } from "../../../components/ui";
import type { Segment } from "../../../types/asr";

export function SegmentsTable({ rows }: { rows: Segment[] }) {
  if (!rows.length) return <EmptyState>No segments yet.</EmptyState>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>Start</th><th>End</th><th>Risk</th><th>Text</th></tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className={row.suspicious ? "suspicious-row" : ""}>
              <td>{row.start}</td>
              <td>{row.end}</td>
              <td>{row.suspicious ? "Review" : "OK"}</td>
              <td>{row.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
