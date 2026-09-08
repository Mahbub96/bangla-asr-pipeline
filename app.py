#!/usr/bin/env python3
"""
Bangla & English Automatic Speech Recognition (ASR) - Interactive Web Interface
Features:
- Live microphone recording & audio upload
- Smart Bilingual Language Routing (Indic phonetics -> Bangla, English -> English)
- Export transcriptions to Plain Text (.txt), Subtitles (.srt, .vtt), and JSON (.json)
- Batch transcription via multi-file upload or server directory path
- Accuracy benchmarking (WER & CER) on ground-truth datasets
- System diagnostics & model status inspect
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR / "scripts"))

from transcribe import get_transcriber, transcribe_file
from evaluate import compute_metrics

# Global model cache to avoid reloading weights on every inference
MODEL_CACHE = {}

def get_system_device_info():
    """Detects available hardware acceleration (CPU or CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"CUDA GPU: {device_name} ({vram_gb:.1f} GB VRAM)"
    except Exception:
        pass
    import os
    cpu_count = os.cpu_count() or 1
    return f"CPU: Multi-Core ({cpu_count} threads, INT8 CTranslate2)"

def load_cached_model(model_name="large-v3-turbo"):
    """Loads and caches model in memory."""
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = get_transcriber(
            model_size=model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(SCRIPT_DIR / "models")
        )
    return MODEL_CACHE[model_name]

# ==============================================================================
# Subtitle & File Export Helpers
# ==============================================================================
def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def generate_srt(segments: list) -> str:
    """Converts Whisper segment objects into standard SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg.get("start", 0.0))
        end = format_timestamp_srt(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def generate_vtt(segments: list) -> str:
    """Converts Whisper segment objects into WebVTT format."""
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = format_timestamp_vtt(seg.get("start", 0.0))
        end = format_timestamp_vtt(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def create_temp_export(content: str, suffix: str) -> str:
    """Writes text content to a temporary file and returns path for download."""
    t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
    t.write(content)
    t.close()
    return t.name

# ==============================================================================
# Tab 1: Single Audio & Microphone Handler
# ==============================================================================
def transcribe_audio_ui(audio_path, model_name, language_choice, beam_size, temperature, initial_prompt, vad_filter):
    """Processes uploaded or recorded audio and returns structured results."""
    if not audio_path:
        return (
            "Please record speech using your microphone or upload an audio file (.wav, .mp3, .flac).",
            "",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    try:
        model = load_cached_model(model_name)
        res = transcribe_file(
            model,
            audio_path,
            language=lang_code,
            beam_size=int(beam_size),
            initial_prompt=initial_prompt if initial_prompt else None,
            vad_filter=vad_filter,
            temperature=float(temperature)
        )

        full_text = res.get("text", "").strip()
        lang_detected = res.get("language", "unknown")
        lang_prob = res.get("language_probability", 1.0)
        duration = res.get("duration_sec", 0.0)
        proc_time = res.get("transcription_time_sec", 0.0)
        speed = res.get("speed_factor", 0.0)

        word_count = len(full_text.split()) if full_text else 0
        char_count = len(full_text)

        flag = "🇧🇩" if "bn" in lang_detected.lower() or "bangla" in lang_detected.lower() else "🇬🇧"

        stats_html = f"""
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; margin-bottom: 8px;">
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 140px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Detected Language</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                    {flag} {lang_detected} <span style="font-size: 0.85rem; font-weight: 400; color: #059669;">({lang_prob:.1%})</span>
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 130px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Audio Duration</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                    ⏱️ {duration:.2f}s
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 140px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Speed Factor</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #059669; margin-top: 2px;">
                    🚀 {speed}x real-time <span style="font-size: 0.8rem; font-weight: 400; color: #64748b;">({proc_time:.2f}s)</span>
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 120px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Length</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                    📊 {word_count} words <span style="font-size: 0.8rem; font-weight: 400; color: #64748b;">({char_count} chars)</span>
                </div>
            </div>
        </div>
        """

        segments = res.get("segments", [])
        segments_data = [
            {
                "Start (s)": s.get("start", 0.0),
                "End (s)": s.get("end", 0.0),
                "Duration (s)": round(s.get("end", 0.0) - s.get("start", 0.0), 2),
                "Transcription": s.get("text", "")
            }
            for s in segments
        ]
        df_segments = pd.DataFrame(segments_data)

        # Generate downloadable files
        txt_path = create_temp_export(full_text, ".txt")
        srt_path = create_temp_export(generate_srt(segments), ".srt")
        vtt_path = create_temp_export(generate_vtt(segments), ".vtt")
        json_path = create_temp_export(json.dumps(res, ensure_ascii=False, indent=2), ".json")

        return (
            full_text,
            stats_html,
            df_segments,
            json.dumps(res, ensure_ascii=False, indent=2),
            gr.update(value=txt_path, visible=True),
            gr.update(value=srt_path, visible=True),
            gr.update(value=vtt_path, visible=True),
            gr.update(value=json_path, visible=True)
        )

    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}"
        return (
            error_msg,
            f"<div style='color: #dc2626; font-weight: 600;'>❌ {error_msg}</div>",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

# ==============================================================================
# Tab 2: Batch Audio Transcription Handler
# ==============================================================================
def batch_transcribe_ui(input_mode, uploaded_files, directory_path, model_name, language_choice, progress=gr.Progress(track_tqdm=True)):
    """Transcribes multiple audio files uploaded via browser or found in server directory."""
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_paths = []

    if input_mode == "Upload Files directly in Browser":
        if not uploaded_files:
            return "Please upload at least one audio file.", None, gr.update(visible=False), gr.update(visible=False)
        for f in uploaded_files:
            p = Path(f.name if hasattr(f, 'name') else str(f))
            if p.suffix.lower() in valid_exts:
                audio_paths.append(p)
    else:
        dir_p = Path(directory_path)
        if not dir_p.exists() or not dir_p.is_dir():
            return f"Directory '{directory_path}' does not exist on the server.", None, gr.update(visible=False), gr.update(visible=False)
        audio_paths = [p for p in dir_p.rglob("*") if p.suffix.lower() in valid_exts]

    if not audio_paths:
        return "No valid audio files found (.wav, .mp3, .flac, .ogg, .m4a).", None, gr.update(visible=False), gr.update(visible=False)

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    model = load_cached_model(model_name)
    results = []
    total_duration = 0.0
    total_proc_time = 0.0

    progress(0, desc="Starting batch transcription...")
    for idx, f in enumerate(audio_paths):
        pct = (idx + 1) / len(audio_paths)
        progress(pct, desc=f"Processing {idx+1}/{len(audio_paths)}: {f.name}")
        try:
            res = transcribe_file(model, f, language=lang_code)
            results.append({
                "File": f.name,
                "Language": res["language"],
                "Confidence": f"{res['language_probability']:.1%}",
                "Duration (s)": res["duration_sec"],
                "Speed": f"{res['speed_factor']}x",
                "Transcription": res["text"]
            })
            total_duration += res["duration_sec"]
            total_proc_time += res["transcription_time_sec"]
        except Exception as err:
            results.append({
                "File": f.name,
                "Language": "Error",
                "Confidence": "N/A",
                "Duration (s)": 0,
                "Speed": "N/A",
                "Transcription": str(err)
            })

    df = pd.DataFrame(results)

    out_csv = create_temp_export(df.to_csv(index=False), ".csv")
    out_json = create_temp_export(json.dumps(results, ensure_ascii=False, indent=2), ".json")

    avg_speed = round(total_duration / total_proc_time, 2) if total_proc_time > 0 else 0
    summary_html = f"""
    <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">
        <div style="background: #ecfdf5; border: 1px solid #6ee7b7; padding: 12px 16px; border-radius: 8px; flex: 1;">
            <span style="color: #065f46; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Total Transcribed</span>
            <div style="font-size: 1.25rem; font-weight: 700; color: #064e3b; margin-top: 2px;">
                ✅ {len(results)} Files
            </div>
        </div>
        <div style="background: #eff6ff; border: 1px solid #93c5fd; padding: 12px 16px; border-radius: 8px; flex: 1;">
            <span style="color: #1e40af; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Total Audio Time</span>
            <div style="font-size: 1.25rem; font-weight: 700; color: #1e3a8a; margin-top: 2px;">
                ⏱️ {total_duration:.1f}s ({total_duration/60:.1f} min)
            </div>
        </div>
        <div style="background: #f0fdf4; border: 1px solid #86efac; padding: 12px 16px; border-radius: 8px; flex: 1;">
            <span style="color: #166534; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Average Speed</span>
            <div style="font-size: 1.25rem; font-weight: 700; color: #14532d; margin-top: 2px;">
                🚀 {avg_speed}x real-time
            </div>
        </div>
    </div>
    """

    return summary_html, df, gr.update(value=out_csv, visible=True), gr.update(value=out_json, visible=True)

# ==============================================================================
# Tab 3: Dataset Benchmark (WER / CER) Handler
# ==============================================================================
def evaluate_dataset_ui(csv_file_upload, metadata_csv_path, audio_dir, model_name, language_choice, progress=gr.Progress(track_tqdm=True)):
    """Calculates WER & CER benchmark on test set."""
    if csv_file_upload is not None:
        csv_p = Path(csv_file_upload.name if hasattr(csv_file_upload, 'name') else str(csv_file_upload))
    else:
        csv_p = Path(metadata_csv_path)

    if not csv_p.exists():
        return f"Error: CSV file '{csv_p}' not found.", "", None, gr.update(visible=False)

    df = pd.read_csv(csv_p)
    audio_col = next((c for c in ["audio_path", "audio", "file_name", "path"] if c in df.columns), None)
    text_col = next((c for c in ["sentence", "transcription", "ground_truth", "text"] if c in df.columns), None)

    if not audio_col or not text_col:
        return f"Error: CSV must have audio path and text columns. Detected columns: {list(df.columns)}", "", None, gr.update(visible=False)

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    try:
        import jiwer
    except ImportError:
        return "Error: jiwer library is required for WER/CER evaluation. Please install it in .venv.", "", None, gr.update(visible=False)

    model = load_cached_model(model_name)
    audio_base = Path(audio_dir) if audio_dir else csv_p.parent

    eval_rows = []
    progress(0, desc="Evaluating test dataset...")

    for idx, (_, row) in enumerate(df.iterrows()):
        progress((idx + 1) / len(df), desc=f"Evaluating sample {idx+1}/{len(df)}...")
        rel_audio = str(row[audio_col]).strip()
        ref_text = str(row[text_col]).strip()

        audio_full = audio_base / rel_audio
        if not audio_full.is_file() and Path(rel_audio).is_file():
            audio_full = Path(rel_audio)
        if not audio_full.is_file() and (audio_base / "test_done" / rel_audio).is_file():
            audio_full = audio_base / "test_done" / rel_audio

        if not audio_full.is_file():
            continue

        try:
            pred = transcribe_file(model, audio_full, language=lang_code)
            hyp_text = pred["text"].strip()
            wer = jiwer.wer(ref_text, hyp_text) if ref_text else 1.0
            cer = jiwer.cer(ref_text, hyp_text) if ref_text else 1.0

            eval_rows.append({
                "File": rel_audio,
                "Ground Truth": ref_text,
                "Prediction": hyp_text,
                "WER": f"{wer:.1%}",
                "CER": f"{cer:.1%}"
            })
        except Exception:
            pass

    if not eval_rows:
        return "No valid audio files referenced in the CSV were found.", "", None, gr.update(visible=False)

    results_df = pd.DataFrame(eval_rows)
    all_refs = [r["Ground Truth"] for r in eval_rows]
    all_hyps = [r["Prediction"] for r in eval_rows]
    overall_wer = jiwer.wer(all_refs, all_hyps)
    overall_cer = jiwer.cer(all_refs, all_hyps)

    wer_color = "#16a34a" if overall_wer < 0.15 else ("#ca8a04" if overall_wer < 0.35 else "#dc2626")
    cer_color = "#16a34a" if overall_cer < 0.10 else ("#ca8a04" if overall_cer < 0.25 else "#dc2626")

    summary_cards = f"""
    <div style="display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 12px;">
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1;">
            <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Evaluated Samples</span>
            <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 4px;">
                📈 {len(results_df)} / {len(df)}
            </div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1;">
            <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Overall Word Error Rate (WER)</span>
            <div style="font-size: 1.5rem; font-weight: 700; color: {wer_color}; margin-top: 4px;">
                🎯 {overall_wer:.2%}
            </div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1;">
            <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Overall Character Error Rate (CER)</span>
            <div style="font-size: 1.5rem; font-weight: 700; color: {cer_color}; margin-top: 4px;">
                🔤 {overall_cer:.2%}
            </div>
        </div>
    </div>
    """

    out_csv = create_temp_export(results_df.to_csv(index=False), ".csv")
    return "Evaluation completed successfully!", summary_cards, results_df, gr.update(value=out_csv, visible=True)

# ==============================================================================
# Tab 4: System Diagnostics Handler
# ==============================================================================
def get_diagnostics():
    """Returns real-time system status, loaded models, and disk cache info."""
    models_dir = SCRIPT_DIR / "models"
    models_found = []
    if models_dir.exists():
        for d in models_dir.glob("models--*"):
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
            models_found.append(f"• **{d.name.replace('models--', '')}**: {size_mb:.1f} MB on disk")

    models_info = "\n".join(models_found) if models_found else "No pre-downloaded models found in `models/`."
    cached_in_ram = list(MODEL_CACHE.keys()) if MODEL_CACHE else ["None (loaded on demand)"]

    return f"""
### 🖥️ Hardware & Execution Environment
- **Compute Device**: `{get_system_device_info()}`
- **Quantization Engine**: `CTranslate2 INT8 (CPU-Optimized)`
- **Python Version**: `{sys.version.split()[0]}`
- **Gradio Version**: `{gr.__version__}`

---

### 🧠 Model Cache Status
- **Models Currently Loaded in RAM**: `{', '.join(cached_in_ram)}`
- **Cached Model Weights on Disk (`models/`)**:
{models_info}

---

### 🌐 Smart Bilingual Routing Status
- **Supported Primary Languages**: Bangla (বাংলা) & English
- **Phonetic Routing**: Automatic rerouting of Indic phonetic confusion (Hindi / Urdu / Assamese) directly to Bangla (`bn`)
- **VAD Silence Trimming**: Enabled (500ms min silence duration)
"""

# ==============================================================================
# Gradio UI Construction & Styling
# ==============================================================================
custom_css = """
/* Responsive Bengali & English Typography */
.bangla-output textarea {
    font-size: 1.22rem !important;
    line-height: 1.85 !important;
    font-family: 'SolaimanLipi', 'Noto Sans Bengali', 'Hind Siliguri', 'Segoe UI', system-ui, sans-serif !important;
    color: #0f172a !important;
    background-color: #f8fafc !important;
    border-radius: 8px !important;
}

/* Custom action buttons styling */
.action-btn {
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.15s ease-in-out;
}
.action-btn:hover {
    transform: translateY(-1px);
}
"""

theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Noto Sans Bengali", "sans-serif"]
)

with gr.Blocks(title="Bangla & English ASR - Whisper") as demo:
    # Header Banner
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%); padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 style="font-size: 1.75rem; font-weight: 800; margin: 0; display: flex; align-items: center; gap: 10px;">
                    🎙️ Bangla & English Speech Recognition (ASR)
                </h1>
                <p style="font-size: 0.95rem; margin: 6px 0 0 0; opacity: 0.9;">
                    High-accuracy speech-to-text powered by OpenAI's Whisper with CTranslate2 INT8 acceleration & Smart Bilingual Routing.
                </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; backdrop-filter: blur(4px);">
                    🇧🇩 বাংলা & 🇬🇧 English
                </span>
                <span style="background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; backdrop-filter: blur(4px);">
                    ⚡ INT8 CTranslate2
                </span>
                <span style="background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; backdrop-filter: blur(4px);">
                    🚀 large-v3-turbo
                </span>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ======================================================================
        # TAB 1: Live Microphone & Single File
        # ======================================================================
        with gr.TabItem("🎯 Live Mic & Audio File Transcription"):
            with gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record Speech or Upload Audio File (.wav, .mp3, .flac, .ogg, .m4a)"
                    )

                    with gr.Accordion("⚙️ Model, Decoding & Language Settings", open=True):
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=["large-v3-turbo", "tiny"],
                                value="large-v3-turbo",
                                label="Whisper Checkpoint",
                                scale=1
                            )
                            lang_dropdown = gr.Dropdown(
                                choices=[
                                    "Auto (Smart Bilingual: Bangla / English)",
                                    "Bangla (বাংলা)",
                                    "English (en)"
                                ],
                                value="Auto (Smart Bilingual: Bangla / English)",
                                label="Language Routing",
                                scale=1
                            )

                        with gr.Row():
                            beam_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Beam Size (Accuracy vs Speed)"
                            )
                            temp_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.1,
                                label="Temperature (0.0 = Deterministic)"
                            )

                        with gr.Row():
                            vad_checkbox = gr.Checkbox(
                                value=True,
                                label="Enable VAD (Voice Activity Detection filter)"
                            )

                        prompt_input = gr.Textbox(
                            label="Context Hints / Initial Vocabulary Prompt (Optional)",
                            placeholder="e.g. বাংলাদেশ, কৃত্রিম বুদ্ধিমত্তা, ঢাকা, Whisper ASR",
                            lines=1
                        )

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                        transcribe_btn = gr.Button("🚀 Transcribe Speech", variant="primary", size="lg", scale=2)

                    # Dynamic Quick Test Samples
                    candidate_samples = [
                        "data/test/audio/test.mp3",
                        "data/test/audio/test_done/test.mp3",
                        "data/test/audio/sample_test1.mp3",
                        "data/test/audio/test_done/sample_test1.mp3",
                        "data/test/audio/sample_test2.mp3",
                        "data/test/audio/test_done/sample_test2.mp3",
                        "data/test/audio/complete_Actor - 46_03-01-06-02-04-03-46.wav",
                        "data/test/audio/test_done/complete_Actor - 46_03-01-06-02-04-03-46.wav",
                    ]
                    available_examples = []
                    seen_names = set()
                    for c_path in candidate_samples:
                        p = Path(c_path)
                        if p.exists() and p.name not in seen_names:
                            seen_names.add(p.name)
                            available_examples.append([str(p), "large-v3-turbo", "Auto (Smart Bilingual: Bangla / English)", 5, 0.0, "", True])

                    if not available_examples:
                        for p in sorted(Path("data/test/audio").rglob("*")):
                            if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"} and p.name not in seen_names:
                                seen_names.add(p.name)
                                available_examples.append([str(p), "large-v3-turbo", "Auto (Smart Bilingual: Bangla / English)", 5, 0.0, "", True])
                                if len(available_examples) >= 4:
                                    break

                    if available_examples:
                        gr.Examples(
                            examples=available_examples,
                            inputs=[audio_input, model_dropdown, lang_dropdown, beam_slider, temp_slider, prompt_input, vad_checkbox],
                            label="📁 Quick Test Samples"
                        )

                # Right Column: Outputs & Export Suite
                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="📝 Transcribed Text",
                        lines=7,
                        placeholder="Transcribed speech will appear here...",
                        elem_classes=["bangla-output"]
                    )

                    metrics_output = gr.HTML()

                    # Export Suite Bar
                    gr.Markdown("##### 💾 Export Transcriptions")
                    with gr.Row():
                        download_txt = gr.DownloadButton("📄 Text (.txt)", visible=False, size="sm")
                        download_srt = gr.DownloadButton("🎬 Subtitles (.srt)", visible=False, size="sm")
                        download_vtt = gr.DownloadButton("🌐 WebVTT (.vtt)", visible=False, size="sm")
                        download_json = gr.DownloadButton("📦 JSON (.json)", visible=False, size="sm")

                    with gr.Accordion("⏱️ Timestamped Segments Breakdown", open=False):
                        segments_table = gr.DataFrame(
                            headers=["Start (s)", "End (s)", "Duration (s)", "Transcription"],
                            label="Individual Speech Segments",
                            wrap=True
                        )

                    with gr.Accordion("🔍 Full JSON Metadata", open=False):
                        json_output = gr.Code(language="json", label="Raw JSON Output")

            transcribe_btn.click(
                fn=transcribe_audio_ui,
                inputs=[audio_input, model_dropdown, lang_dropdown, beam_slider, temp_slider, prompt_input, vad_checkbox],
                outputs=[output_text, metrics_output, segments_table, json_output, download_txt, download_srt, download_vtt, download_json]
            )

            clear_btn.click(
                fn=lambda: (None, "", "", None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
                outputs=[audio_input, output_text, metrics_output, segments_table, json_output, download_txt, download_srt, download_vtt, download_json]
            )

        # ======================================================================
        # TAB 2: Batch Audio Transcription
        # ======================================================================
        with gr.TabItem("📂 Batch Audio Transcription"):
            gr.Markdown("""
            ### Batch Transcribe Audio Files
            Process multiple audio files concurrently. You can drag and drop multiple files directly from your browser, or specify a server directory path.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    batch_mode = gr.Radio(
                        choices=["Upload Files directly in Browser", "Specify Server Directory Path"],
                        value="Upload Files directly in Browser",
                        label="Batch Source Mode"
                    )

                    batch_files_upload = gr.File(
                        file_count="multiple",
                        file_types=["audio"],
                        label="Drop Audio Files Here (.wav, .mp3, .flac, .ogg, .m4a)",
                        visible=True
                    )

                    batch_dir_input = gr.Textbox(
                        value="data/test/audio",
                        label="Directory Path on Server",
                        visible=False
                    )

                    batch_mode.change(
                        fn=lambda mode: (gr.update(visible=mode == "Upload Files directly in Browser"), gr.update(visible=mode != "Upload Files directly in Browser")),
                        inputs=[batch_mode],
                        outputs=[batch_files_upload, batch_dir_input]
                    )

                    with gr.Row():
                        batch_model = gr.Dropdown(
                            choices=["large-v3-turbo", "tiny"],
                            value="large-v3-turbo",
                            label="Whisper Model"
                        )
                        batch_lang = gr.Dropdown(
                            choices=[
                                "Auto (Smart Bilingual: Bangla / English)",
                                "Bangla (বাংলা)",
                                "English (en)"
                            ],
                            value="Auto (Smart Bilingual: Bangla / English)",
                            label="Language Mode"
                        )

                    batch_btn = gr.Button("⚡ Start Batch Transcription", variant="primary", size="lg")

                with gr.Column(scale=1):
                    batch_status_html = gr.HTML()
                    with gr.Row():
                        batch_download_csv = gr.DownloadButton("📥 Download CSV Report", visible=False)
                        batch_download_json = gr.DownloadButton("📥 Download JSON Results", visible=False)

            batch_table = gr.DataFrame(
                headers=["File", "Language", "Confidence", "Duration (s)", "Speed", "Transcription"],
                label="Batch Results",
                wrap=True
            )

            batch_btn.click(
                fn=batch_transcribe_ui,
                inputs=[batch_mode, batch_files_upload, batch_dir_input, batch_model, batch_lang],
                outputs=[batch_status_html, batch_table, batch_download_csv, batch_download_json]
            )

        # ======================================================================
        # TAB 3: Dataset Benchmark (WER & CER)
        # ======================================================================
        with gr.TabItem("📊 Benchmark & Evaluate (WER / CER)"):
            gr.Markdown("""
            ### Accuracy Benchmark Tool
            Evaluate speech recognition accuracy using **Word Error Rate (WER)** and **Character Error Rate (CER)** against a test dataset with ground truth references.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    eval_csv_upload = gr.File(
                        label="Upload Metadata CSV (Optional: overrides path below)",
                        file_types=[".csv"]
                    )
                    eval_csv_input = gr.Textbox(
                        value="data/test/metadata.csv",
                        label="Metadata CSV File Path on Server"
                    )
                    eval_audio_dir = gr.Textbox(
                        value="data/test/audio",
                        label="Audio Directory Path"
                    )
                    with gr.Row():
                        eval_model = gr.Dropdown(
                            choices=["large-v3-turbo", "tiny"],
                            value="large-v3-turbo",
                            label="Whisper Model"
                        )
                        eval_lang = gr.Dropdown(
                            choices=[
                                "Auto (Smart Bilingual: Bangla / English)",
                                "Bangla (বাংলা)",
                                "English (en)"
                            ],
                            value="Auto (Smart Bilingual: Bangla / English)",
                            label="Language Mode"
                        )

                    eval_btn = gr.Button("📈 Run Benchmark & Compute WER/CER", variant="primary", size="lg")

                with gr.Column(scale=1):
                    eval_status = gr.Markdown()
                    eval_metrics_html = gr.HTML()
                    eval_download_csv = gr.DownloadButton("📥 Download Sample-by-Sample Error Report (CSV)", visible=False)

            eval_results_table = gr.DataFrame(
                headers=["File", "Ground Truth", "Prediction", "WER", "CER"],
                label="Sample Predictions vs Ground Truth",
                wrap=True
            )

            eval_btn.click(
                fn=evaluate_dataset_ui,
                inputs=[eval_csv_upload, eval_csv_input, eval_audio_dir, eval_model, eval_lang],
                outputs=[eval_status, eval_metrics_html, eval_results_table, eval_download_csv]
            )

        # ======================================================================
        # TAB 4: System Diagnostics & Health Check
        # ======================================================================
        with gr.TabItem("🖥️ System Diagnostics & Cache"):
            diag_output = gr.Markdown(value=get_diagnostics())
            refresh_diag_btn = gr.Button("🔄 Refresh Diagnostics", variant="secondary")
            refresh_diag_btn.click(fn=get_diagnostics, outputs=[diag_output])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=custom_css)
