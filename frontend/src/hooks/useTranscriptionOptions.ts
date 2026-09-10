import { useState } from "react";

export function useTranscriptionOptions() {
  const [model, setModel] = useState("large-v3-turbo");
  const [language, setLanguage] = useState("auto");
  const [beam, setBeam] = useState(5);
  const [temperature, setTemperature] = useState(0);
  const [prompt, setPrompt] = useState("");
  const [vad, setVad] = useState(true);

  function appendOptions(form: FormData) {
    form.append("model_name", model);
    form.append("language", language);
    form.append("beam_size", String(beam));
    form.append("temperature", String(temperature));
    form.append("initial_prompt", prompt);
    form.append("vad_filter", String(vad));
  }

  return {
    options: { model, language, beam, temperature, prompt, vad },
    setters: { setModel, setLanguage, setBeam, setTemperature, setPrompt, setVad },
    appendOptions
  };
}
