import { ASR_LANGUAGES, ASR_MODELS } from "../../../constants/asr";
import { Field } from "../../../components/ui";
import type { useTranscriptionOptions } from "../../../hooks/useTranscriptionOptions";

type TranscriptionOptionsHook = ReturnType<typeof useTranscriptionOptions>;

export function DecodeOptionsPanel({ options, setters }: Pick<TranscriptionOptionsHook, "options" | "setters">) {
  return (
    <details className="options-disclosure">
      <summary>Model and decoding options</summary>
      <div className="option-grid">
        <Field label="Model">
          <select value={options.model} onChange={(event) => setters.setModel(event.target.value)}>
            {ASR_MODELS.map((item) => <option key={item}>{item}</option>)}
          </select>
        </Field>
        <Field label="Language">
          <select value={options.language} onChange={(event) => setters.setLanguage(event.target.value)}>
            {ASR_LANGUAGES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
        </Field>
        <Field label="Beam">
          <input type="number" min={1} max={10} value={options.beam} onChange={(event) => setters.setBeam(Number(event.target.value))} />
        </Field>
        <Field label="Temperature">
          <input type="number" min={0} max={1} step={0.1} value={options.temperature} onChange={(event) => setters.setTemperature(Number(event.target.value))} />
        </Field>
        <Field label="Context hints">
          <input value={options.prompt} onChange={(event) => setters.setPrompt(event.target.value)} placeholder="বাংলাদেশ, ঢাকা, ASR" />
        </Field>
        <label className="check">
          <input type="checkbox" checked={options.vad} onChange={(event) => setters.setVad(event.target.checked)} />
          VAD silence trimming
        </label>
      </div>
    </details>
  );
}
