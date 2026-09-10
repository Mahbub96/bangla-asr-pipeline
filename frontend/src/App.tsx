import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Activity,
  BarChart3,
  Braces,
  Clock,
  Cpu,
  Download,
  FileAudio,
  FolderOpen,
  ListTree,
  Mic,
  Play,
  RefreshCw,
  Square,
  Terminal,
  Trash2,
  Upload,
  Waves
} from "lucide-react";
import { Button, Field, Panel, StatusPill } from "./components/ui";
import { checkMicrophone, type MicState } from "./lib/media";
import { exportUrl, getJson, postForm, postJson, type JobResponse, type JobSnapshot } from "./lib/api";

const models = ["large-v3-turbo", "tiny"];
const languages = [
  { label: "Auto Bangla / English", value: "auto" },
  { label: "Bangla", value: "bn" },
  { label: "English", value: "en" }
];

type Tab = "live" | "batch" | "benchmark" | "training" | "diagnostics";
type OutputView = "transcript" | "segments" | "raw";

function App() {
  const [tab, setTab] = useState<Tab>("live");
  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Bangla & English ASR Studio</h1>
          <p>React + FastAPI workstation console for Whisper inference, evaluation, and training.</p>
        </div>
        <div className="topbar-pills">
          <StatusPill tone="good">Local API</StatusPill>
          <StatusPill>Whisper</StatusPill>
          <StatusPill>LoRA Ready</StatusPill>
        </div>
      </header>
      <nav className="tabs" aria-label="Studio sections">
        <TabButton active={tab === "live"} onClick={() => setTab("live")} icon={<Mic size={16} />} label="Live Mic & Audio" />
        <TabButton active={tab === "batch"} onClick={() => setTab("batch")} icon={<FolderOpen size={16} />} label="Batch" />
        <TabButton active={tab === "benchmark"} onClick={() => setTab("benchmark")} icon={<BarChart3 size={16} />} label="Benchmark" />
        <TabButton active={tab === "training"} onClick={() => setTab("training")} icon={<Terminal size={16} />} label="Training" />
        <TabButton active={tab === "diagnostics"} onClick={() => setTab("diagnostics")} icon={<Cpu size={16} />} label="Diagnostics" />
      </nav>
      {tab === "live" && <LiveTranscription />}
      {tab === "batch" && <BatchTranscription />}
      {tab === "benchmark" && <Benchmark />}
      {tab === "training" && <Training />}
      {tab === "diagnostics" && <Diagnostics />}
    </main>
  );
}

function TabButton({ active, icon, label, onClick }: { active: boolean; icon: ReactNode; label: string; onClick: () => void }) {
  return (
    <button className={`tab-btn ${active ? "active" : ""}`} onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function useOptions() {
  const [model, setModel] = useState("large-v3-turbo");
  const [language, setLanguage] = useState("auto");
  const [beam, setBeam] = useState(5);
  const [temperature, setTemperature] = useState(0);
  const [prompt, setPrompt] = useState("");
  const [vad, setVad] = useState(true);
  const appendOptions = (form: FormData) => {
    form.append("model_name", model);
    form.append("language", language);
    form.append("beam_size", String(beam));
    form.append("temperature", String(temperature));
    form.append("initial_prompt", prompt);
    form.append("vad_filter", String(vad));
  };
  const controls = (
    <div className="option-grid">
      <Field label="Model">
        <select value={model} onChange={(event) => setModel(event.target.value)}>
          {models.map((item) => <option key={item}>{item}</option>)}
        </select>
      </Field>
      <Field label="Language">
        <select value={language} onChange={(event) => setLanguage(event.target.value)}>
          {languages.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </select>
      </Field>
      <Field label="Beam">
        <input type="number" min={1} max={10} value={beam} onChange={(event) => setBeam(Number(event.target.value))} />
      </Field>
      <Field label="Temperature">
        <input type="number" min={0} max={1} step={0.1} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
      </Field>
      <Field label="Context hints">
        <input value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="বাংলাদেশ, ঢাকা, ASR" />
      </Field>
      <label className="check">
        <input type="checkbox" checked={vad} onChange={(event) => setVad(event.target.checked)} />
        VAD silence trimming
      </label>
    </div>
  );
  return { appendOptions, controls, model, language };
}

function LiveTranscription() {
  const { appendOptions, controls } = useOptions();
  const [micState, setMicState] = useState<MicState>("checking");
  const [file, setFile] = useState<File | null>(null);
  const [recorded, setRecorded] = useState<Blob | null>(null);
  const [recordedUrl, setRecordedUrl] = useState("");
  const [recordedDuration, setRecordedDuration] = useState(0);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [inputLevel, setInputLevel] = useState(0);
  const [waveform, setWaveform] = useState<number[]>([]);
  const [outputView, setOutputView] = useState<OutputView>("transcript");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState<any>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationRef = useRef<number | null>(null);
  const recordingStartedAt = useRef(0);

  useEffect(() => {
    checkMicrophone(false).then(setMicState);
    return () => {
      stopMeter();
      if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    };
  }, [recordedUrl]);

  function formatDuration(seconds: number) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const tenths = Math.floor((seconds % 1) * 10);
    return `${mins}:${secs.toString().padStart(2, "0")}.${tenths}`;
  }

  function stopMeter() {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setInputLevel(0);
  }

  function startMeter(stream: MediaStream) {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const context = new AudioCtx();
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    context.createMediaStreamSource(stream).connect(analyser);
    audioContextRef.current = context;
    const data = new Uint8Array(analyser.fftSize);

    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (const sample of data) {
        const normalized = (sample - 128) / 128;
        sum += normalized * normalized;
      }
      const rms = Math.sqrt(sum / data.length);
      setInputLevel(Math.min(1, rms * 5));
      setRecordingSeconds((performance.now() - recordingStartedAt.current) / 1000);
      animationRef.current = requestAnimationFrame(tick);
    };
    tick();
  }

  async function buildWaveform(blob: Blob) {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const context = new AudioCtx();
    try {
      const buffer = await context.decodeAudioData(await blob.arrayBuffer());
      const channel = buffer.getChannelData(0);
      const bars = 120;
      const block = Math.max(1, Math.floor(channel.length / bars));
      const nextWaveform = Array.from({ length: bars }, (_, index) => {
        let peak = 0;
        const start = index * block;
        for (let i = start; i < Math.min(start + block, channel.length); i += 1) {
          peak = Math.max(peak, Math.abs(channel[i]));
        }
        return Math.max(0.04, Math.min(1, peak));
      });
      setWaveform(nextWaveform);
      setRecordedDuration(buffer.duration);
    } finally {
      await context.close();
    }
  }

  function setRecordingBlob(blob: Blob) {
    if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    const url = URL.createObjectURL(blob);
    setRecorded(blob);
    setRecordedUrl(url);
    buildWaveform(blob).catch(() => setWaveform([]));
  }

  function resetLiveState() {
    setFile(null);
    setRecorded(null);
    setRecordedDuration(0);
    setRecordingSeconds(0);
    setWaveform([]);
    setPayload(null);
    setError("");
    setOutputView("transcript");
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl);
      setRecordedUrl("");
    }
  }

  async function handleFile(fileValue: File | null) {
    setFile(fileValue);
    setPayload(null);
    setOutputView("transcript");
    if (!fileValue) return;
    setRecorded(null);
    setRecordedDuration(0);
    setRecordedUrl("");
    setWaveform([]);
    if (fileValue.type.startsWith("audio/")) {
      await buildWaveform(fileValue).catch(() => setWaveform([]));
    }
  }

  async function startRecording() {
    setError("");
    setPayload(null);
    setFile(null);
    setRecorded(null);
    setRecordedDuration(0);
    setWaveform([]);
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl);
      setRecordedUrl("");
    }
    const state = await checkMicrophone(true);
    if (state !== "ready") {
      setMicState(state);
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    chunks.current = [];
    recordingStartedAt.current = performance.now();
    setRecordingSeconds(0);
    recorder.current = new MediaRecorder(stream);
    recorder.current.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.current.push(event.data);
    };
    recorder.current.onstop = () => {
      const duration = (performance.now() - recordingStartedAt.current) / 1000;
      const blob = new Blob(chunks.current, { type: "audio/webm" });
      stopMeter();
      setRecordedDuration(duration);
      setRecordingBlob(blob);
      setMicState("stopped");
    };
    recorder.current.start();
    startMeter(stream);
    setMicState("recording");
  }

  function stopRecording() {
    recorder.current?.stop();
  }

  async function submit() {
    const source = file ?? (recorded ? new File([recorded], "recording.webm", { type: "audio/webm" }) : null);
    if (!source) return setError("Record speech or choose an audio file first.");
    setBusy(true);
    setError("");
    setOutputView("transcript");
    try {
      const form = new FormData();
      form.append("audio", source);
      appendOptions(form);
      setPayload(await postForm("/api/transcribe", form));
    } catch (exc: any) {
      setError(exc.message);
    } finally {
      setBusy(false);
    }
  }

  const result = payload?.result;
  const sourceReady = Boolean(recorded || file);
  const activeDuration = micState === "recording" ? recordingSeconds : recordedDuration;
  return (
    <div className="workspace live-grid">
      <Panel title="Live Mic & Audio" action={<MicBadge state={micState} />}>
        <div className={`recording-console ${micState === "recording" ? "is-recording" : ""}`}>
          <div className="recording-primary">
            <div className="recording-state">
              <span className="recording-dot" />
              <span>{micState === "recording" ? "Recording in progress" : sourceReady ? "Audio source ready" : "Waiting for audio"}</span>
            </div>
            <div className="recording-time">
              <Clock size={18} />
              {formatDuration(activeDuration)}
            </div>
          </div>

          <div className="live-meter" aria-label="Live microphone input level">
            {Array.from({ length: 24 }, (_, index) => {
              const lit = index / 23 < inputLevel;
              return <span key={index} className={lit ? "active" : ""} />;
            })}
          </div>

          <Waveform bars={waveform} active={micState === "recording"} level={inputLevel} />

          {recordedUrl && <audio className="audio-preview" src={recordedUrl} controls />}

          <div className="recording-actions">
            <Button onClick={micState === "recording" ? stopRecording : startRecording} className={micState === "recording" ? "danger" : "primary"}>
              {micState === "recording" ? <Square size={16} /> : <Mic size={16} />}
              {micState === "recording" ? "Stop Recording" : "Start Recording"}
            </Button>
            <Button onClick={() => checkMicrophone(true).then(setMicState)} disabled={micState === "recording"}>
              <RefreshCw size={16} />
              Check Mic
            </Button>
            <Button className="primary" onClick={submit} disabled={busy || !sourceReady || micState === "recording"}>
              <Play size={16} />
              {busy ? "Transcribing" : "Transcribe Audio"}
            </Button>
            <Button onClick={resetLiveState} disabled={micState === "recording"}>
              <Trash2 size={16} />
              Clear
            </Button>
          </div>
        </div>

        <div className="source-panel">
          <label className="upload-zone compact-upload">
            <Upload size={22} />
            <span>{file ? file.name : "Choose an existing audio file"}</span>
            <input type="file" accept="audio/*" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} />
          </label>
          <div className="source-summary">
            <Info label="Source" value={file ? "Uploaded file" : recorded ? "Microphone recording" : "No audio selected"} />
            <Info label="Duration" value={sourceReady ? formatDuration(activeDuration) : "0:00.0"} />
          </div>
        </div>

        <details className="options-disclosure">
          <summary>Model and decoding options</summary>
          {controls}
        </details>
        {error && <p className="error">{error}</p>}
      </Panel>

      <Panel title="Backend Output" action={result && <ExportLinks exports={payload.exports} />}>
        <div className="output-toolbar" role="tablist" aria-label="Transcription output views">
          <button className={outputView === "transcript" ? "active" : ""} onClick={() => setOutputView("transcript")}>
            <FileAudio size={16} />
            Raw transcript
          </button>
          <button className={outputView === "segments" ? "active" : ""} onClick={() => setOutputView("segments")}>
            <ListTree size={16} />
            Time segments
          </button>
          <button className={outputView === "raw" ? "active" : ""} onClick={() => setOutputView("raw")}>
            <Braces size={16} />
            API JSON
          </button>
        </div>

        {!result && (
          <div className="structured-empty">
            <Waves size={32} />
            <strong>No backend result yet</strong>
            <span>Record or upload audio, then transcribe to populate transcript, time segments, metadata, and export links.</span>
          </div>
        )}

        {result && outputView === "transcript" && (
          <>
            <textarea className="transcript enhanced" value={result.text ?? ""} readOnly />
            <Metrics result={result} />
          </>
        )}
        {result && outputView === "segments" && <Segments rows={result.segments ?? []} />}
        {result && outputView === "raw" && <pre className="json-view">{JSON.stringify(payload, null, 2)}</pre>}
      </Panel>
    </div>
  );
}

function Waveform({ bars, active, level }: { bars: number[]; active: boolean; level: number }) {
  if (!active && bars.length === 0) {
    return (
      <div className="waveform empty-waveform" aria-label="No audio waveform loaded">
        <div className="waveform-ruler">
          <span>0:00</span>
          <span>timeline</span>
          <span>clip end</span>
        </div>
        <div className="waveform-placeholder">
          <Waves size={28} />
          <span>Waveform appears after recording or upload</span>
        </div>
      </div>
    );
  }
  const displayBars = bars.length
    ? bars
    : Array.from({ length: 96 }, (_, index) => {
        if (!active) return 0.08;
        const phase = Math.sin(index * 0.55 + Date.now() / 130);
        return Math.max(0.05, Math.min(1, level * (0.45 + Math.abs(phase) * 0.9)));
      });
  return (
    <div className={`waveform ${active ? "live" : ""}`} aria-label={active ? "Live recording waveform" : "Recorded audio waveform"}>
      <div className="waveform-ruler">
        <span>0:00</span>
        <span>timeline</span>
        <span>clip end</span>
      </div>
      <div className="waveform-bars">
        {displayBars.map((bar, index) => (
          <span key={index} style={{ height: `${Math.max(8, bar * 92)}%` }} />
        ))}
      </div>
    </div>
  );
}

function MicBadge({ state }: { state: MicState }) {
  const tone = state === "ready" || state === "stopped" ? "good" : state === "recording" ? "live" : state === "checking" ? "neutral" : "bad";
  const labels: Record<MicState, string> = {
    checking: "Checking mic",
    unsupported: "Mic unsupported",
    insecure: "Use localhost or HTTPS",
    "permission-denied": "Permission blocked",
    "no-device": "No mic detected",
    ready: "Mic ready",
    recording: "Recording",
    stopped: "Recorded",
    failed: "Mic unavailable"
  };
  return <StatusPill tone={tone}>{labels[state]}</StatusPill>;
}

function Metrics({ result }: { result: any }) {
  return (
    <div className="metrics">
      <span>Language <strong>{result.language}</strong></span>
      <span>Confidence <strong>{Math.round(result.language_probability * 100)}%</strong></span>
      <span>Duration <strong>{result.duration_sec}s</strong></span>
      <span>Speed <strong>{result.speed_factor}x</strong></span>
    </div>
  );
}

function Segments({ rows }: { rows: any[] }) {
  if (!rows.length) return <div className="empty">No segments yet.</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Start</th><th>End</th><th>Text</th></tr></thead>
        <tbody>{rows.map((row, index) => <tr key={index}><td>{row.start}</td><td>{row.end}</td><td>{row.text}</td></tr>)}</tbody>
      </table>
    </div>
  );
}

function ExportLinks({ exports }: { exports: Record<string, string> }) {
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

function useJobStream() {
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [error, setError] = useState("");
  function attach(response: JobResponse) {
    setError("");
    const events = new EventSource(response.events_url);
    events.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setJob(data.job);
      if (["done"].includes(data.type)) events.close();
    };
    events.onerror = () => {
      events.close();
      setError("Progress stream disconnected. Refresh job status if needed.");
    };
  }
  async function cancel() {
    if (!job) return;
    setJob(await postJson(`/api/jobs/${job.id}/cancel`, {}));
  }
  return { job, setJob, error, setError, attach, cancel };
}

function BatchTranscription() {
  const { appendOptions, controls } = useOptions();
  const [files, setFiles] = useState<FileList | null>(null);
  const [directory, setDirectory] = useState("data/test/audio");
  const [mode, setMode] = useState<"files" | "directory">("files");
  const { job, error, setError, attach, cancel } = useJobStream();
  async function submit() {
    try {
      const form = new FormData();
      appendOptions(form);
      if (mode === "files" && files) Array.from(files as FileList).forEach((file) => form.append("files", file));
      if (mode === "directory") form.append("directory_path", directory);
      attach(await postForm("/api/batch/jobs", form));
    } catch (exc: any) {
      setError(exc.message);
    }
  }
  return (
    <JobLayout title="Batch Transcription" job={job} error={error} onCancel={cancel}>
      <div className="segmented">
        <button className={mode === "files" ? "active" : ""} onClick={() => setMode("files")}>Upload Files</button>
        <button className={mode === "directory" ? "active" : ""} onClick={() => setMode("directory")}>Server Directory</button>
      </div>
      {mode === "files" ? <input type="file" multiple accept="audio/*" onChange={(event) => setFiles(event.target.files)} /> : (
        <Field label="Directory path"><input value={directory} onChange={(event) => setDirectory(event.target.value)} /></Field>
      )}
      {controls}
      <Button className="primary" onClick={submit}><FileAudio size={16} /> Start Batch</Button>
    </JobLayout>
  );
}

function Benchmark() {
  const [csv, setCsv] = useState<File | null>(null);
  const [csvPath, setCsvPath] = useState("data/test/metadata.csv");
  const [audioDir, setAudioDir] = useState("data/test/audio");
  const [model, setModel] = useState("large-v3-turbo");
  const [language, setLanguage] = useState("auto");
  const { job, error, setError, attach, cancel } = useJobStream();
  async function submit() {
    try {
      const form = new FormData();
      if (csv) form.append("csv_file", csv);
      form.append("metadata_csv_path", csvPath);
      form.append("audio_dir", audioDir);
      form.append("model_name", model);
      form.append("language", language);
      attach(await postForm("/api/evaluate/jobs", form));
    } catch (exc: any) {
      setError(exc.message);
    }
  }
  return (
    <JobLayout title="Benchmark WER / CER" job={job} error={error} onCancel={cancel}>
      <input type="file" accept=".csv" onChange={(event) => setCsv(event.target.files?.[0] ?? null)} />
      <div className="option-grid">
        <Field label="Metadata CSV"><input value={csvPath} onChange={(event) => setCsvPath(event.target.value)} /></Field>
        <Field label="Audio directory"><input value={audioDir} onChange={(event) => setAudioDir(event.target.value)} /></Field>
        <Field label="Model"><select value={model} onChange={(event) => setModel(event.target.value)}>{models.map((item) => <option key={item}>{item}</option>)}</select></Field>
        <Field label="Language"><select value={language} onChange={(event) => setLanguage(event.target.value)}>{languages.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></Field>
      </div>
      <Button className="primary" onClick={submit}><Activity size={16} /> Run Benchmark</Button>
    </JobLayout>
  );
}

function Training() {
  const { job, error, setError, attach, cancel } = useJobStream();
  const [config, setConfig] = useState({
    model_name_or_path: "openai/whisper-large-v3-turbo",
    language: "bengali",
    task: "transcribe",
    train_csv: "data/train/metadata.csv",
    train_audio: "data/train/audio",
    val_csv: "data/val/metadata.csv",
    val_audio: "data/val/audio",
    output_dir: "./checkpoints/whisper_bangla_lora",
    batch_size: 8,
    eval_batch_size: 8,
    gradient_accumulation_steps: 2,
    learning_rate: "1e-4",
    num_epochs: 5,
    finetune_mode: "lora",
    precision: "fp16"
  });
  function set(key: string, value: string | number) {
    setConfig((current) => ({ ...current, [key]: value }));
  }
  async function submit() {
    try {
      attach(await postJson("/api/train/jobs", config));
    } catch (exc: any) {
      setError(exc.message);
    }
  }
  return (
    <JobLayout title="Batched Training" job={job} error={error} onCancel={cancel} logsOnly>
      <div className="option-grid">
        {Object.entries(config).map(([key, value]) => (
          <Field key={key} label={key.replaceAll("_", " ")}>
            <input value={value} onChange={(event) => set(key, typeof value === "number" ? Number(event.target.value) : event.target.value)} />
          </Field>
        ))}
      </div>
      <Button className="primary" onClick={submit}><Terminal size={16} /> Launch Training</Button>
    </JobLayout>
  );
}

function JobLayout({ title, children, job, error, onCancel, logsOnly = false }: { title: string; children: ReactNode; job: JobSnapshot | null; error: string; onCancel: () => void; logsOnly?: boolean }) {
  const rows = Array.isArray(job?.result) ? job?.result : job?.result?.rows;
  return (
    <div className="workspace two-col">
      <Panel title={title}>{children}{error && <p className="error">{error}</p>}</Panel>
      <Panel title="Progress" action={job && <Button onClick={onCancel} disabled={!["queued", "running"].includes(job.status)}><Square size={16} /> Cancel</Button>}>
        {job ? (
          <>
            <div className="job-head">
              <StatusPill tone={job.status === "completed" ? "good" : job.status === "failed" ? "bad" : "live"}>{job.status}</StatusPill>
              <span>{job.message}</span>
            </div>
            <progress value={job.progress} max={1} />
            <pre className="console">{job.logs.join("\n")}</pre>
            <ExportLinks exports={job.exports} />
            {!logsOnly && rows && <ResultTable rows={rows} />}
          </>
        ) : <div className="empty">Start a job to see progress, logs, and downloads.</div>}
      </Panel>
    </div>
  );
}

function ResultTable({ rows }: { rows: any[] }) {
  const columns = useMemo(() => Object.keys(rows[0] ?? {}), [rows]);
  if (!rows.length) return null;
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{columns.map((col) => <th key={col}>{col}</th>)}</tr></thead>
        <tbody>{rows.map((row, index) => <tr key={index}>{columns.map((col) => <td key={col}>{String(row[col])}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function Diagnostics() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  async function load() {
    try {
      setError("");
      setData(await getJson("/api/diagnostics"));
    } catch (exc: any) {
      setError(exc.message);
    }
  }
  useEffect(() => { load(); }, []);
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
            <Info label="Disk Models" value={data.disk_models.map((item: any) => `${item.name} ${item.size_mb}MB`).join(", ") || "None"} />
          </div>
        ) : <div className="empty">Loading diagnostics.</div>}
      </Panel>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return <div className="info"><span>{label}</span><strong>{value}</strong></div>;
}

export default App;
