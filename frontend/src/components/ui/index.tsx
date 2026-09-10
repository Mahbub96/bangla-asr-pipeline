import { Download, Upload } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { exportUrl } from "../../lib/api";
import type { ExportMap } from "../../types/asr";

export function Button({ className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`btn ${className}`} {...props} />;
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return <span className="spinner" role="status" aria-label={label} />;
}

export function Panel({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

export function RangeField({
  label,
  value,
  min,
  max,
  step = 1,
  suffix = "",
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field range-field">
      <span className="range-label">
        {label}
        <strong>{value}{suffix}</strong>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      <span className="range-scale">
        <small>{min}{suffix}</small>
        <small>{max}{suffix}</small>
      </span>
    </label>
  );
}

export function StatusPill({ tone = "neutral", children }: { tone?: "neutral" | "good" | "warn" | "bad" | "live"; children: ReactNode }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

export function Tabs({ children, label }: { children: ReactNode; label: string }) {
  return (
    <nav className="tabs" aria-label={label}>
      {children}
    </nav>
  );
}

export function SegmentedControl<T extends string>({
  value,
  options,
  onChange
}: {
  value: T;
  options: Array<{ label: string; value: T }>;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented">
      {options.map((option) => (
        <button key={option.value} className={value === option.value ? "active" : ""} onClick={() => onChange(option.value)}>
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function MetricGrid({ items }: { items: Array<{ label: string; value: ReactNode }> }) {
  return (
    <div className="metrics">
      {items.map((item) => (
        <span key={item.label}>
          {item.label} <strong>{item.value}</strong>
        </span>
      ))}
    </div>
  );
}

export function EmptyState({ children, className = "empty" }: { children: ReactNode; className?: string }) {
  return <div className={className}>{children}</div>;
}

export function DataTable({ columns, rows }: { columns: string[]; rows: Array<Record<string, unknown>> }) {
  if (!rows.length) return null;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>{columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ConsoleLog({ logs }: { logs: string[] }) {
  return <pre className="console">{logs.join("\n")}</pre>;
}

export function FileDropzone({
  label,
  accept,
  multiple,
  onChange
}: {
  label: ReactNode;
  accept?: string;
  multiple?: boolean;
  onChange: (files: FileList | null) => void;
}) {
  return (
    <label className="upload-zone compact-upload">
      <Upload size={22} />
      <span>{label}</span>
      <input type="file" accept={accept} multiple={multiple} onChange={(event) => onChange(event.target.files)} />
    </label>
  );
}

export function ExportLinks({ exports }: { exports?: ExportMap }) {
  if (!exports || !Object.keys(exports).length) return null;
  return (
    <div className="export-links">
      {Object.entries(exports).map(([key, path]) => (
        <a key={key} href={exportUrl(path)}>
          <Download size={14} />
          {key.toUpperCase()}
        </a>
      ))}
    </div>
  );
}

export function Info({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="info">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
