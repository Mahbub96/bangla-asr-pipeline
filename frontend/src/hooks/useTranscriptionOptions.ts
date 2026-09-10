import { useState } from "react";

function clampNumber(value: number, min: number, max: number) {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export function useTranscriptionOptions() {
  const [model, setModel] = useState("large-v3-turbo");
  const [language, setLanguage] = useState("auto");
  const [beam, setBeam] = useState(5);
  const [temperature, setTemperature] = useState(0);
  const [prompt, setPrompt] = useState("");
  const [vad, setVad] = useState(true);
  const [profile, setProfile] = useState("auto");
  const [chunkLength, setChunkLength] = useState(30);
  const [vadAggressiveness, setVadAggressiveness] = useState("medium");
  const [conditionOnPreviousText, setConditionOnPreviousText] = useState(true);
  const [repetitionGuard, setRepetitionGuard] = useState(true);
  const [hotwords, setHotwords] = useState("");
  const [outputScript, setOutputScript] = useState("native");

  function appendOptions(form: FormData) {
    form.append("model_name", model);
    form.append("language", language);
    form.append("beam_size", String(beam));
    form.append("temperature", String(temperature));
    form.append("initial_prompt", prompt);
    form.append("vad_filter", String(vad));
    form.append("profile", profile);
    form.append("chunk_length", String(chunkLength));
    form.append("vad_aggressiveness", vadAggressiveness);
    form.append("condition_on_previous_text", String(conditionOnPreviousText));
    form.append("repetition_guard", String(repetitionGuard));
    form.append("hotwords", hotwords);
    form.append("output_script", outputScript);
  }

  return {
    options: {
      model,
      language,
      beam,
      temperature,
      prompt,
      vad,
      profile,
      chunkLength,
      vadAggressiveness,
      conditionOnPreviousText,
      repetitionGuard,
      hotwords,
      outputScript
    },
    setters: {
      setModel,
      setLanguage,
      setBeam: (value: number) => setBeam(clampNumber(value, 1, 10)),
      setTemperature: (value: number) => setTemperature(clampNumber(value, 0, 1)),
      setPrompt,
      setVad,
      setProfile,
      setChunkLength: (value: number) => setChunkLength(clampNumber(value, 0, 120)),
      setVadAggressiveness,
      setConditionOnPreviousText,
      setRepetitionGuard,
      setHotwords,
      setOutputScript
    },
    appendOptions
  };
}
