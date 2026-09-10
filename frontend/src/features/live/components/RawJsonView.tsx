export function RawJsonView({ payload }: { payload: unknown }) {
  return <pre className="json-view">{JSON.stringify(payload, null, 2)}</pre>;
}
