import { ACCURACY_PROFILES, ASR_LANGUAGES, ASR_MODELS, VAD_AGGRESSIVENESS } from "../../../constants/asr";
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
        <Field label="Accuracy profile">
          <select value={options.profile} onChange={(event) => setters.setProfile(event.target.value)}>
            {ACCURACY_PROFILES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
        </Field>
        <Field label="Beam">
          <input type="number" min={1} max={10} value={options.beam} onChange={(event) => setters.setBeam(Number(event.target.value))} />
        </Field>
        <Field label="Chunk seconds">
          <input type="number" min={0} max={120} value={options.chunkLength} onChange={(event) => setters.setChunkLength(Number(event.target.value))} />
        </Field>
        <Field label="VAD strength">
          <select value={options.vadAggressiveness} onChange={(event) => setters.setVadAggressiveness(event.target.value)}>
            {VAD_AGGRESSIVENESS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
        </Field>
        <Field label="Temperature">
          <input type="number" min={0} max={1} step={0.1} value={options.temperature} onChange={(event) => setters.setTemperature(Number(event.target.value))} />
        </Field>
        <Field label="Context hints">
          <input value={options.prompt} onChange={(event) => setters.setPrompt(event.target.value)} placeholder="বাংলাদেশ, ঢাকা, ASR" />
        </Field>
        <Field label="Hotwords">
          <input value={options.hotwords} onChange={(event) => setters.setHotwords(event.target.value)} placeholder="সংসদ কৃষক সার সরকার" />
        </Field>
        <label className="check">
          <input type="checkbox" checked={options.vad} onChange={(event) => setters.setVad(event.target.checked)} />
          VAD silence trimming
        </label>
        <label className="check">
          <input type="checkbox" checked={options.conditionOnPreviousText} onChange={(event) => setters.setConditionOnPreviousText(event.target.checked)} />
          Condition on previous text
        </label>
        <label className="check">
          <input type="checkbox" checked={options.repetitionGuard} onChange={(event) => setters.setRepetitionGuard(event.target.checked)} />
          Repetition guard
        </label>
      </div>
    </details>
  );
}
