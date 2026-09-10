import { DataTable, EmptyState } from "../../../components/ui";
import type { Segment } from "../../../types/asr";

export function SegmentsTable({ rows }: { rows: Segment[] }) {
  if (!rows.length) return <EmptyState>No segments yet.</EmptyState>;
  return <DataTable columns={["start", "end", "text"]} rows={rows as unknown as Array<Record<string, unknown>>} />;
}
