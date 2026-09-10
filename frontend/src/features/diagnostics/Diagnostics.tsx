import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { Button, EmptyState, Info, Panel } from "../../components/ui";
import { getJson } from "../../lib/api";
import type { DiagnosticsPayload } from "../../types/asr";

export function Diagnostics() {
  const [data, setData] = useState<DiagnosticsPayload | null>(null);
  const [error, setError] = useState("");

  async function load() {
    try {
      setError("");
      setData(await getJson("/api/diagnostics"));
    } catch (exc: any) {
      setError(exc.message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="workspace">
      <Panel title="System Diagnostics" action={<Button onClick={load}><RefreshCw size={16} /> Refresh</Button>}>
        {error && <p className="error">{error}</p>}
        {data ? (
          <div className="diagnostics">
            <Info label="Compute" value={data.compute} />
            <Info label="Operating System" value={data.os} />
            <Info label="Python" value={data.python} />
            <Info label="FFmpeg" value={data.ffmpeg ?? "Not found"} />
            <Info label="Loaded Models" value={data.loaded_models.length ? JSON.stringify(data.loaded_models) : "None"} />
            <Info label="Disk Models" value={data.disk_models.map((item) => `${item.name} ${item.size_mb}MB`).join(", ") || "None"} />
          </div>
        ) : (
          <EmptyState>Loading diagnostics.</EmptyState>
        )}
      </Panel>
    </div>
  );
}
