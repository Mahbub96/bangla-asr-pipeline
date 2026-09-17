export type { ExportMap, JobResponse, JobSnapshot, Segment, TranscriptionResult } from "../types/asr";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function errorMessage(response: Response): Promise<string> {
  const error = await response.json().catch(() => ({ detail: response.statusText }));
  if (Array.isArray(error.detail)) {
    return error.detail
      .map((item: any) => {
        const location = Array.isArray(item.loc) ? item.loc.join(".") : "";
        return `${location ? `${location}: ` : ""}${item.msg}`;
      })
      .join("\n");
  }
  return error.detail ?? response.statusText ?? "Request failed";
}

export async function postForm<T>(url: string, form: FormData): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, { method: "POST", body: form });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return response.json();
}

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return response.json();
}

export async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export function exportUrl(path: string): string {
  return `${API_BASE}${path}`;
}
