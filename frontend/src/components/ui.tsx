import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Button({ className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`btn ${className}`} {...props} />;
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

export function StatusPill({ tone = "neutral", children }: { tone?: "neutral" | "good" | "warn" | "bad" | "live"; children: ReactNode }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

