#!/usr/bin/env python3
"""
Bangla & English Automatic Speech Recognition (ASR) - Interactive Web Interface
Features:
- Live microphone recording & audio upload with streaming recognition
- Instant Stop / Cancel controls across all operations
- Client-side one-click Copy to Clipboard
- Smart Bilingual Language Routing (Indic phonetics -> Bangla, English -> English)
- Export transcriptions to Plain Text (.txt), Subtitles (.srt, .vtt), and JSON (.json)
- Streamed batch transcription via multi-file upload or server directory path
- Accuracy benchmarking (WER & CER) on ground-truth datasets
- Comprehensive Batched Training / Fine-Tuning GUI with ALL model & training hyperparameters,
  dynamic model-adaptive presets, live console streaming, and stop controls.
- System diagnostics & model status inspector
- Full display height (100vh) single-screen dashboard layout without unnecessary window scrolling.
"""

import json
import os
import subprocess
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

# Active training subprocess tracker
ACTIVE_TRAIN_PROC = None

# Model Architecture Adaptive Presets
MODEL_PRESETS = {
    "openai/whisper-large-v3-turbo": {
        "batch_size": 8,
        "eval_batch_size": 8,
        "grad_accum": 2,
        "learning_rate": "1e-4",
        "lora_r": 32,
        "lora_alpha": 64,
        "target_modules": "q_proj,v_proj",
        "finetune_mode": "LoRA (Parameter-Efficient PEFT) - Recommended",
        "precision": "FP16 Mixed Precision",
        "optim": "adamw_torch",
        "description": "⚡ **Whisper Large-v3-Turbo** (809M params, 4 decoder layers): High accuracy with fast decoding. Recommended batch size 8 with LoRA for 16GB-24GB VRAM."
    },
    "openai/whisper-large-v3": {
        "batch_size": 4,
        "eval_batch_size": 4,
        "grad_accum": 4,
        "learning_rate": "5e-5",
        "lora_r": 32,
        "lora_alpha": 64,
        "target_modules": "q_proj,v_proj",
        "finetune_mode": "LoRA (Parameter-Efficient PEFT) - Recommended",
        "precision": "FP16 Mixed Precision",
        "optim": "adamw_torch",
        "description": "🏆 **Whisper Large-v3** (1550M params, 32 decoder layers): Maximum model capacity. Requires smaller batch size (4) and grad accumulation (4) to prevent CUDA out-of-memory."
    },
    "openai/whisper-medium": {
        "batch_size": 4,
        "eval_batch_size": 4,
        "grad_accum": 4,
        "learning_rate": "1e-4",
        "lora_r": 32,
        "lora_alpha": 64,
        "target_modules": "q_proj,v_proj",
        "finetune_mode": "LoRA (Parameter-Efficient PEFT) - Recommended",
        "precision": "FP16 Mixed Precision",
        "optim": "adamw_torch",
        "description": "⚖️ **Whisper Medium** (769M params, 24 decoder layers): Strong bilingual capabilities with moderate VRAM requirements."
    },
    "openai/whisper-small": {
        "batch_size": 8,
        "eval_batch_size": 8,
        "grad_accum": 2,
        "learning_rate": "1e-4",
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": "q_proj,v_proj",
        "finetune_mode": "LoRA (Parameter-Efficient PEFT) - Recommended",
        "precision": "FP16 Mixed Precision",
        "optim": "adamw_torch",
        "description": "📦 **Whisper Small** (244M params): Light memory footprint, comfortably fine-tunes on 8GB-12GB consumer GPUs."
    },
    "openai/whisper-base": {
        "batch_size": 16,
        "eval_batch_size": 16,
        "grad_accum": 1,
        "learning_rate": "1e-4",
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": "q_proj,v_proj",
        "finetune_mode": "Full Model Fine-Tuning",
        "precision": "FP32 (Standard)",
        "optim": "adamw_torch",
        "description": "🚀 **Whisper Base** (74M params): Highly responsive. Can be full-finetuned directly on consumer GPUs or CPUs."
    },
    "openai/whisper-tiny": {
        "batch_size": 16,
        "eval_batch_size": 16,
        "grad_accum": 1,
        "learning_rate": "1e-4",
        "lora_r": 8,
        "lora_alpha": 16,
        "target_modules": "q_proj,v_proj",
        "finetune_mode": "Full Model Fine-Tuning",
        "precision": "FP32 (Standard)",
        "optim": "adamw_torch",
        "description": "🌱 **Whisper Tiny** (39M params): Ultra-lightweight. Perfect for fast pipeline debugging and testing mini-batches on CPU."
    }
}

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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def generate_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg.get("start", 0.0))
        end = format_timestamp_srt(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def generate_vtt(segments: list) -> str:
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = format_timestamp_vtt(seg.get("start", 0.0))
        end = format_timestamp_vtt(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def create_temp_export(content: str, suffix: str) -> str:
    t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
    t.write(content)
    t.close()
    return t.name

# ==============================================================================
# Tab 1: Streaming Single Audio & Microphone Handler with Stop Support
# ==============================================================================
def transcribe_audio_streaming(audio_path, model_name, language_choice, beam_size, temperature, initial_prompt, vad_filter):
    if not audio_path:
        yield (
            "Please record speech using your microphone or upload an audio file (.wav, .mp3, .flac).",
            "<div style='color: #64748b; font-size: 0.82rem;'>Waiting for audio input...</div>",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
        return

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    yield (
        "",
        "<div style='background: #eff6ff; color: #1d4ed8; padding: 6px 10px; border-radius: 6px; border: 1px solid #bfdbfe; font-size: 0.85rem; font-weight: 600;'>⏳ Loading Whisper model & initializing audio stream...</div>",
        None,
        None,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False)
    )

    try:
        model = load_cached_model(model_name)
        audio_p = Path(audio_path)
        if not audio_p.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        INDIC_LANGS = {"hi", "bn", "ur", "as", "mr", "ne", "gu", "pa", "or", "sa"}
        target_lang = lang_code
        detected_lang = "bn (Bangla)" if lang_code == "bn" else ("en (English)" if lang_code == "en" else None)
        lang_prob = 1.0

        if lang_code == "auto":
            from faster_whisper import decode_audio
            try:
                audio_data = decode_audio(str(audio_p))
                raw_lang, raw_prob, all_probs = model.detect_language(audio_data)
                lang_dict = dict(all_probs)
                indic_score = sum(lang_dict.get(l, 0.0) for l in INDIC_LANGS)
                en_score = lang_dict.get("en", 0.0)

                if indic_score >= en_score:
                    target_lang = "bn"
                    detected_lang = "bn (Bangla)"
                    lang_prob = indic_score
                else:
                    target_lang = "en"
                    detected_lang = "en (English)"
                    lang_prob = en_score
            except Exception:
                target_lang = None

        transcribe_kwargs = {
            "beam_size": int(beam_size),
            "language": target_lang,
            "vad_filter": vad_filter,
            "temperature": float(temperature)
        }
        if vad_filter:
            transcribe_kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)
        if initial_prompt and initial_prompt.strip():
            transcribe_kwargs["initial_prompt"] = initial_prompt.strip()

        start_time = time.time()
        segments_gen, info = model.transcribe(str(audio_p), **transcribe_kwargs)

        if detected_lang is None:
            detected_lang = info.language
            lang_prob = info.language_probability
        total_duration = info.duration

        flag = "🇧🇩" if "bn" in str(detected_lang).lower() or "bangla" in str(detected_lang).lower() else "🇬🇧"

        collected_segments = []
        full_text_list = []

        for seg in segments_gen:
            collected_segments.append({
                "Start (s)": round(seg.start, 2),
                "End (s)": round(seg.end, 2),
                "Duration (s)": round(seg.end - seg.start, 2),
                "Transcription": seg.text.strip()
            })
            full_text_list.append(seg.text.strip())

            current_text = " ".join(full_text_list)
            elapsed = time.time() - start_time

            streaming_status = f"""
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 10px; margin-bottom: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: 700; color: #0284c7; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
                        <span class="pulsing-dot"></span> ⏳ Transcribing live ({seg.end:.1f}s / {total_duration:.1f}s)...
                    </span>
                    <span style="font-size: 0.80rem; color: #64748b; font-weight: 600;">
                        {flag} {detected_lang} ({lang_prob:.1%})
                    </span>
                </div>
                <div style="background: #e2e8f0; border-radius: 3px; height: 5px; overflow: hidden;">
                    <div style="background: #0284c7; width: {min(100, int((seg.end / max(1.0, total_duration)) * 100))}%; height: 100%; transition: width 0.2s ease;"></div>
                </div>
            </div>
            """

            df_curr = pd.DataFrame(collected_segments)
            yield (
                current_text,
                streaming_status,
                df_curr,
                json.dumps({"current_progress_sec": seg.end, "segments_count": len(collected_segments)}, indent=2),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )

        elapsed_total = time.time() - start_time
        final_speed = round(total_duration / elapsed_total, 2) if elapsed_total > 0 else 0
        final_text = " ".join(full_text_list).strip()
        word_count = len(final_text.split()) if final_text else 0
        char_count = len(final_text)

        completion_html = f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; margin-bottom: 4px;">
            <div style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; border: 1px solid #cbd5e1; flex: 1; min-width: 110px;">
                <span style="font-size: 0.70rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Language</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 1px;">
                    {flag} {detected_lang} <span style="font-size: 0.75rem; font-weight: 400; color: #059669;">({lang_prob:.1%})</span>
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; border: 1px solid #cbd5e1; flex: 1; min-width: 90px;">
                <span style="font-size: 0.70rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Duration</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 1px;">
                    ⏱️ {total_duration:.1f}s
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; border: 1px solid #cbd5e1; flex: 1; min-width: 110px;">
                <span style="font-size: 0.70rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Speed</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #059669; margin-top: 1px;">
                    🚀 {final_speed}x <span style="font-size: 0.75rem; font-weight: 400; color: #64748b;">({elapsed_total:.1f}s)</span>
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; border: 1px solid #cbd5e1; flex: 1; min-width: 90px;">
                <span style="font-size: 0.70rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Words</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 1px;">
                    📊 {word_count} w <span style="font-size: 0.75rem; font-weight: 400; color: #64748b;">({char_count} c)</span>
                </div>
            </div>
        </div>
        """

        df_final = pd.DataFrame(collected_segments)
        full_result_dict = {
            "file": str(audio_p),
            "duration_sec": round(total_duration, 2),
            "transcription_time_sec": round(elapsed_total, 2),
            "speed_factor": final_speed,
            "language": detected_lang,
            "language_probability": round(lang_prob, 4),
            "text": final_text,
            "segments": collected_segments
        }

        txt_p = create_temp_export(final_text, ".txt")
        srt_p = create_temp_export(generate_srt(collected_segments), ".srt")
        vtt_p = create_temp_export(generate_vtt(collected_segments), ".vtt")
        json_p = create_temp_export(json.dumps(full_result_dict, ensure_ascii=False, indent=2), ".json")

        yield (
            final_text,
            completion_html,
            df_final,
            json.dumps(full_result_dict, ensure_ascii=False, indent=2),
            gr.update(value=txt_p, visible=True),
            gr.update(value=srt_p, visible=True),
            gr.update(value=vtt_p, visible=True),
            gr.update(value=json_p, visible=True)
        )

    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}"
        yield (
            error_msg,
            f"<div style='color: #dc2626; font-weight: 600; padding: 6px 10px; background: #fee2e2; border-radius: 6px; font-size: 0.85rem;'>❌ {error_msg}</div>",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def on_single_transcribe_stop():
    return "<div style='color: #c2410c; font-weight: 600; padding: 6px 10px; background: #fff7ed; border-radius: 6px; border: 1px solid #fed7aa; font-size: 0.85rem;'>🛑 Transcription stopped by user. Ready for new audio.</div>"

def on_single_clear():
    return (
        None,
        "",
        "<div style='color: #64748b; font-size: 0.82rem;'>All inputs cleared. Ready for new audio.</div>",
        None,
        None,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False)
    )

# ==============================================================================
# Tab 2: Streaming Batch Audio Handler with Stop Support
# ==============================================================================
def batch_transcribe_streaming(input_mode, uploaded_files, directory_path, model_name, language_choice):
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_paths = []

    if input_mode == "Upload Files directly in Browser":
        if not uploaded_files:
            yield "Please upload at least one audio file.", None, gr.update(visible=False), gr.update(visible=False)
            return
        for f in uploaded_files:
            p = Path(f.name if hasattr(f, 'name') else str(f))
            if p.suffix.lower() in valid_exts:
                audio_paths.append(p)
    else:
        dir_p = Path(directory_path)
        if not dir_p.exists() or not dir_p.is_dir():
            yield f"Directory '{directory_path}' does not exist on the server.", None, gr.update(visible=False), gr.update(visible=False)
            return
        audio_paths = [p for p in dir_p.rglob("*") if p.suffix.lower() in valid_exts]

    if not audio_paths:
        yield "No valid audio files found (.wav, .mp3, .flac, .ogg, .m4a).", None, gr.update(visible=False), gr.update(visible=False)
        return

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    yield (
        f"<div style='background: #eff6ff; color: #1d4ed8; padding: 6px 10px; border-radius: 6px; border: 1px solid #bfdbfe; font-size: 0.85rem; font-weight: 600;'>⏳ Initializing batch processing for {len(audio_paths)} files...</div>",
        None,
        gr.update(visible=False),
        gr.update(visible=False)
    )

    model = load_cached_model(model_name)
    results = []
    total_duration = 0.0
    total_proc_time = 0.0

    for idx, f in enumerate(audio_paths):
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

        df_curr = pd.DataFrame(results)
        avg_speed = round(total_duration / total_proc_time, 2) if total_proc_time > 0 else 0

        out_csv = create_temp_export(df_curr.to_csv(index=False), ".csv")
        out_json = create_temp_export(json.dumps(results, ensure_ascii=False, indent=2), ".json")

        running_html = f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;">
            <div style="background: #ecfdf5; border: 1px solid #6ee7b7; padding: 6px 12px; border-radius: 6px; flex: 1;">
                <span style="color: #065f46; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Progress</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #064e3b; margin-top: 1px;">
                    ✅ {idx + 1} / {len(audio_paths)} Files
                </div>
            </div>
            <div style="background: #eff6ff; border: 1px solid #93c5fd; padding: 6px 12px; border-radius: 6px; flex: 1;">
                <span style="color: #1e40af; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Audio Time</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #1e3a8a; margin-top: 1px;">
                    ⏱️ {total_duration:.1f}s ({total_duration/60:.1f}m)
                </div>
            </div>
            <div style="background: #f0fdf4; border: 1px solid #86efac; padding: 6px 12px; border-radius: 6px; flex: 1;">
                <span style="color: #166534; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Avg Speed</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #14532d; margin-top: 1px;">
                    🚀 {avg_speed}x real-time
                </div>
            </div>
        </div>
        """

        yield (
            running_html,
            df_curr,
            gr.update(value=out_csv, visible=True),
            gr.update(value=out_json, visible=True)
        )

def on_batch_stop():
    return "<div style='color: #c2410c; font-weight: 600; padding: 6px 10px; background: #fff7ed; border-radius: 6px; border: 1px solid #fed7aa; font-size: 0.85rem;'>🛑 Batch processing stopped by user. Files completed so far are displayed below.</div>"

# ==============================================================================
# Tab 3: Streaming Dataset Benchmark (WER / CER) with Stop Support
# ==============================================================================
def evaluate_dataset_streaming(csv_file_upload, metadata_csv_path, audio_dir, model_name, language_choice):
    if csv_file_upload is not None:
        csv_p = Path(csv_file_upload.name if hasattr(csv_file_upload, 'name') else str(csv_file_upload))
    else:
        csv_p = Path(metadata_csv_path)

    if not csv_p.exists():
        yield f"Error: CSV file '{csv_p}' not found.", "", None, gr.update(visible=False)
        return

    df = pd.read_csv(csv_p)
    audio_col = next((c for c in ["audio_path", "audio", "file_name", "path"] if c in df.columns), None)
    text_col = next((c for c in ["sentence", "transcription", "ground_truth", "text"] if c in df.columns), None)

    if not audio_col or not text_col:
        yield f"Error: CSV must have audio path and text columns. Detected columns: {list(df.columns)}", "", None, gr.update(visible=False)
        return

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    try:
        import jiwer
    except ImportError:
        yield "Error: jiwer library is required for WER/CER evaluation. Please install it in .venv.", "", None, gr.update(visible=False)
        return

    yield (
        "Starting benchmark evaluation...",
        f"<div style='background: #eff6ff; color: #1d4ed8; padding: 6px 10px; border-radius: 6px; border: 1px solid #bfdbfe; font-size: 0.85rem; font-weight: 600;'>⏳ Initializing model and preparing {len(df)} dataset rows...</div>",
        None,
        gr.update(visible=False)
    )

    model = load_cached_model(model_name)
    audio_base = Path(audio_dir) if audio_dir else csv_p.parent

    eval_rows = []

    for idx, (_, row) in enumerate(df.iterrows()):
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
            continue

        results_df = pd.DataFrame(eval_rows)
        all_refs = [r["Ground Truth"] for r in eval_rows]
        all_hyps = [r["Prediction"] for r in eval_rows]
        running_wer = jiwer.wer(all_refs, all_hyps)
        running_cer = jiwer.cer(all_refs, all_hyps)

        wer_color = "#16a34a" if running_wer < 0.15 else ("#ca8a04" if running_wer < 0.35 else "#dc2626")
        cer_color = "#16a34a" if running_cer < 0.10 else ("#ca8a04" if running_cer < 0.25 else "#dc2626")

        summary_cards = f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 12px; flex: 1;">
                <span style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Evaluated Samples</span>
                <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 1px;">
                    📈 {len(results_df)} / {len(df)}
                </div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 12px; flex: 1;">
                <span style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Running WER</span>
                <div style="font-size: 1.15rem; font-weight: 700; color: {wer_color}; margin-top: 1px;">
                    🎯 {running_wer:.2%}
                </div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 12px; flex: 1;">
                <span style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Running CER</span>
                <div style="font-size: 1.15rem; font-weight: 700; color: {cer_color}; margin-top: 1px;">
                    🔤 {running_cer:.2%}
                </div>
            </div>
        </div>
        """

        out_csv = create_temp_export(results_df.to_csv(index=False), ".csv")
        yield (
            f"Evaluating: {len(results_df)} / {len(df)} completed...",
            summary_cards,
            results_df,
            gr.update(value=out_csv, visible=True)
        )

def on_eval_stop():
    return "<div style='color: #c2410c; font-weight: 600; padding: 6px 10px; background: #fff7ed; border-radius: 6px; border: 1px solid #fed7aa; font-size: 0.85rem;'>🛑 Benchmark stopped by user. Results preserved below.</div>"

# ==============================================================================
# Tab 4: Batched Audio Training / Fine-Tuning Manager
# ==============================================================================
def check_training_environment():
    checks = []
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            checks.append(f"🟢 **Compute Hardware**: NVIDIA CUDA GPU `{dev}` ({mem:.1f} GB VRAM) ready.")
        else:
            checks.append("🟡 **Compute Hardware**: No CUDA GPU found (running in CPU mode). CPU can test small mini-batches, but fine-tuning `large-v3-turbo` requires a GPU (≥16GB VRAM recommended).")
    except Exception as e:
        checks.append(f"🔴 **Compute Hardware**: {e}")

    missing = []
    for pkg in ["transformers", "datasets", "peft", "accelerate", "evaluate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        checks.append(f"⚠️ **Missing Packages**: `{', '.join(missing)}` (Run `pip install -r requirements_gpu.txt`)")
    else:
        checks.append("🟢 **All Training Packages Installed**: `transformers`, `datasets`, `peft`, `accelerate`, `evaluate`.")

    return "\n\n".join(checks)

def validate_training_dataset_gui(train_csv, train_audio, val_csv, val_audio):
    report = []
    for name, csv_path, audio_dir in [("Train Dataset", train_csv, train_audio), ("Validation Dataset", val_csv, val_audio)]:
        p_csv = Path(csv_path)
        if not p_csv.is_file():
            report.append(f"❌ **{name}**: CSV not found at `{csv_path}`")
            continue

        try:
            df = pd.read_csv(p_csv)
            audio_col = next((c for c in ["audio_path", "audio", "file_name", "path"] if c in df.columns), None)
            text_col = next((c for c in ["sentence", "transcription", "ground_truth", "text"] if c in df.columns), None)
            if not audio_col or not text_col:
                report.append(f"❌ **{name}**: CSV missing audio/text column. Columns detected: `{list(df.columns)}`")
                continue

            audio_base = Path(audio_dir)
            existing = 0
            for item in df[audio_col]:
                target = audio_base / str(item)
                if target.is_file() or Path(str(item)).is_file():
                    existing += 1

            missing = len(df) - existing
            status_icon = "✅" if missing == 0 else ("⚠️" if existing > 0 else "❌")
            report.append(f"{status_icon} **{name}**: `{existing} / {len(df)}` audio files verified ({missing} missing) in `{audio_dir}`.")
        except Exception as err:
            report.append(f"❌ **{name}**: Error reading CSV: {str(err)}")

    return "\n\n".join(report)

def apply_model_preset(model_name):
    preset = MODEL_PRESETS.get(model_name, {
        "batch_size": 8,
        "eval_batch_size": 8,
        "grad_accum": 2,
        "learning_rate": "1e-4",
        "lora_r": 32,
        "lora_alpha": 64,
        "target_modules": "q_proj,v_proj",
        "finetune_mode": "LoRA (Parameter-Efficient PEFT) - Recommended",
        "precision": "FP16 Mixed Precision",
        "optim": "adamw_torch",
        "description": f"Custom model `{model_name}`."
    })

    eff_text = f"💡 **Effective Batch Size**: `{preset['batch_size']} × {preset['grad_accum']} = {preset['batch_size'] * preset['grad_accum']} samples per step`"

    return (
        preset["batch_size"],
        preset["eval_batch_size"],
        preset["grad_accum"],
        preset["learning_rate"],
        preset["lora_r"],
        preset["lora_alpha"],
        preset["target_modules"],
        preset["finetune_mode"],
        preset["precision"],
        preset["optim"],
        preset["description"],
        eff_text
    )

def build_cli_command(
    model_name, language, task,
    train_csv, train_audio, val_csv, val_audio, output_dir, num_proc,
    finetune_mode, lora_r, lora_alpha, lora_dropout, lora_target_modules,
    batch_size, eval_batch_size, grad_accum, grad_ckpt, precision, dataloader_workers,
    optim, learning_rate, lr_scheduler, warmup_steps, weight_decay, max_grad_norm,
    epochs, max_steps, eval_steps, save_steps, logging_steps, save_limit, best_metric, report_to,
    gen_max_len, gen_beams
):
    cmd = [
        "python3 scripts/train_whisper.py",
        f"--model_name_or_path \"{model_name}\"",
        f"--language \"{language}\"",
        f"--task \"{task}\"",
        f"--train_csv \"{train_csv}\"",
        f"--train_audio \"{train_audio}\"",
        f"--val_csv \"{val_csv}\"",
        f"--val_audio \"{val_audio}\"",
        f"--output_dir \"{output_dir}\"",
        f"--num_proc {int(num_proc)}",
        f"--batch_size {int(batch_size)}",
        f"--eval_batch_size {int(eval_batch_size)}",
        f"--gradient_accumulation_steps {int(grad_accum)}",
        f"--learning_rate {learning_rate}",
        f"--optim \"{optim}\"",
        f"--lr_scheduler_type \"{lr_scheduler}\"",
        f"--warmup_steps {int(warmup_steps)}",
        f"--weight_decay {float(weight_decay)}",
        f"--max_grad_norm {float(max_grad_norm)}",
        f"--num_epochs {int(epochs)}",
        f"--max_steps {int(max_steps)}",
        f"--eval_steps {int(eval_steps)}",
        f"--save_steps {int(save_steps)}",
        f"--logging_steps {int(logging_steps)}",
        f"--save_total_limit {int(save_limit)}",
        f"--metric_for_best_model \"{best_metric}\"",
        f"--report_to \"{report_to}\"",
        f"--generation_max_length {int(gen_max_len)}",
        f"--generation_num_beams {int(gen_beams)}",
        f"--dataloader_num_workers {int(dataloader_workers)}"
    ]

    if grad_ckpt:
        cmd.append("--gradient_checkpointing")
    if "LoRA" in finetune_mode:
        cmd.append("--use_lora")
        cmd.append(f"--lora_r {int(lora_r)}")
        cmd.append(f"--lora_alpha {int(lora_alpha)}")
        cmd.append(f"--lora_dropout {float(lora_dropout)}")
        cmd.append(f"--lora_target_modules \"{lora_target_modules}\"")
    elif "QLoRA" in finetune_mode:
        cmd.append("--use_qlora")
        cmd.append(f"--lora_r {int(lora_r)}")
        cmd.append(f"--lora_alpha {int(lora_alpha)}")
        cmd.append(f"--lora_dropout {float(lora_dropout)}")
        cmd.append(f"--lora_target_modules \"{lora_target_modules}\"")

    if precision == "FP16 Mixed Precision":
        cmd.append("--fp16")
    elif precision == "BF16 (Ampere/Ada)":
        cmd.append("--bf16")

    return " \\\n    ".join(cmd)

def start_training_gui(
    model_name, language, task,
    train_csv, train_audio, val_csv, val_audio, output_dir, num_proc,
    finetune_mode, lora_r, lora_alpha, lora_dropout, lora_target_modules,
    batch_size, eval_batch_size, grad_accum, grad_ckpt, precision, dataloader_workers,
    optim, learning_rate, lr_scheduler, warmup_steps, weight_decay, max_grad_norm,
    epochs, max_steps, eval_steps, save_steps, logging_steps, save_limit, best_metric, report_to,
    gen_max_len, gen_beams
):
    global ACTIVE_TRAIN_PROC

    if ACTIVE_TRAIN_PROC and ACTIVE_TRAIN_PROC.poll() is None:
        yield "⚠️ A training run is already in progress. Please abort it first.", "<div style='color: #ea580c;'>Training already active.</div>"
        return

    missing = [pkg for pkg in ["transformers", "datasets", "peft", "accelerate"] if not _is_pkg_installed(pkg)]
    if missing:
        error_txt = f"❌ Missing required packages for training: {', '.join(missing)}\nPlease run: pip install -r requirements_gpu.txt\n"
        yield error_txt, f"<div style='color: #dc2626; font-weight: 600;'>{error_txt}</div>"
        return

    if not Path(train_csv).is_file():
        yield f"❌ Train CSV '{train_csv}' does not exist.", "<div style='color: #dc2626;'>Train CSV missing</div>"
        return

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "scripts" / "train_whisper.py"),
        "--model_name_or_path", model_name,
        "--language", language,
        "--task", task,
        "--train_csv", train_csv,
        "--train_audio", train_audio,
        "--val_csv", val_csv,
        "--val_audio", val_audio,
        "--output_dir", output_dir,
        "--num_proc", str(int(num_proc)),
        "--batch_size", str(int(batch_size)),
        "--eval_batch_size", str(int(eval_batch_size)),
        "--gradient_accumulation_steps", str(int(grad_accum)),
        "--learning_rate", str(learning_rate),
        "--optim", optim,
        "--lr_scheduler_type", lr_scheduler,
        "--warmup_steps", str(int(warmup_steps)),
        "--weight_decay", str(float(weight_decay)),
        "--max_grad_norm", str(float(max_grad_norm)),
        "--num_epochs", str(int(epochs)),
        "--max_steps", str(int(max_steps)),
        "--eval_steps", str(int(eval_steps)),
        "--save_steps", str(int(save_steps)),
        "--logging_steps", str(int(logging_steps)),
        "--save_total_limit", str(int(save_limit)),
        "--metric_for_best_model", best_metric,
        "--report_to", report_to,
        "--generation_max_length", str(int(gen_max_len)),
        "--generation_num_beams", str(int(gen_beams)),
        "--dataloader_num_workers", str(int(dataloader_workers))
    ]

    if grad_ckpt:
        cmd.append("--gradient_checkpointing")
    if "LoRA" in finetune_mode:
        cmd.append("--use_lora")
        cmd.extend(["--lora_r", str(int(lora_r)), "--lora_alpha", str(int(lora_alpha)), "--lora_dropout", str(float(lora_dropout)), "--lora_target_modules", str(lora_target_modules)])
    elif "QLoRA" in finetune_mode:
        cmd.append("--use_qlora")
        cmd.extend(["--lora_r", str(int(lora_r)), "--lora_alpha", str(int(lora_alpha)), "--lora_dropout", str(float(lora_dropout)), "--lora_target_modules", str(lora_target_modules)])

    if precision == "FP16 Mixed Precision":
        cmd.append("--fp16")
    elif precision == "BF16 (Ampere/Ada)":
        cmd.append("--bf16")

    effective_batch = int(batch_size) * int(grad_accum)
    log_lines = [
        "=" * 70,
        f"🚀 INITIATING WHISPER BATCHED FINE-TUNING",
        "=" * 70,
        f"• Base Model          : {model_name}",
        f"• Target Language     : {language} (task: {task})",
        f"• Per-Device Batch    : {int(batch_size)} (Eval: {int(eval_batch_size)})",
        f"• Gradient Accum Steps: {int(grad_accum)}",
        f"• Effective Batch Size: {effective_batch}",
        f"• Optimizer           : {optim} (lr: {learning_rate}, scheduler: {lr_scheduler})",
        f"• Total Epochs        : {int(epochs)} (max_steps: {int(max_steps)})",
        f"• Fine-Tuning Method  : {finetune_mode}",
        f"• Precision           : {precision}",
        f"• Checkpoint Saving   : every {int(save_steps)} steps (eval every {int(eval_steps)} steps)",
        f"• Output Directory    : {output_dir}",
        "=" * 70,
        "Launching training process...\n"
    ]

    status_html = f"<div style='background: #eff6ff; color: #1e40af; padding: 6px 10px; border-radius: 6px; border: 1px solid #93c5fd; font-size: 0.85rem; font-weight: 600;'>⏳ Training running (Effective Batch Size: {effective_batch})...</div>"
    yield "\n".join(log_lines), status_html

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        ACTIVE_TRAIN_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(SCRIPT_DIR),
            env=env
        )

        for line in iter(ACTIVE_TRAIN_PROC.stdout.readline, ""):
            log_lines.append(line.rstrip())
            if len(log_lines) > 300:
                log_lines = log_lines[-300:]
            yield "\n".join(log_lines), status_html

        ACTIVE_TRAIN_PROC.stdout.close()
        code = ACTIVE_TRAIN_PROC.wait()
        ACTIVE_TRAIN_PROC = None

        if code == 0:
            log_lines.append(f"\n🎉 TRAINING FINISHED SUCCESSFULLY! Checkpoints stored in: {output_dir}")
            final_status = "<div style='background: #ecfdf5; color: #065f46; padding: 6px 10px; border-radius: 6px; border: 1px solid #6ee7b7; font-size: 0.85rem; font-weight: 600;'>✅ Training completed successfully!</div>"
        else:
            log_lines.append(f"\n⚠️ Process exited with return code: {code}")
            final_status = f"<div style='background: #fff1f2; color: #9f1239; padding: 6px 10px; border-radius: 6px; border: 1px solid #fecdd3; font-size: 0.85rem; font-weight: 600;'>⚠️ Training stopped or failed (exit code {code}).</div>"

        yield "\n".join(log_lines), final_status

    except Exception as e:
        ACTIVE_TRAIN_PROC = None
        err_str = f"\n❌ Failed to execute training subprocess: {str(e)}"
        log_lines.append(err_str)
        yield "\n".join(log_lines), f"<div style='color: #dc2626;'>{err_str}</div>"

def stop_training_gui():
    global ACTIVE_TRAIN_PROC
    if ACTIVE_TRAIN_PROC and ACTIVE_TRAIN_PROC.poll() is None:
        try:
            ACTIVE_TRAIN_PROC.terminate()
            time.sleep(1)
            if ACTIVE_TRAIN_PROC.poll() is None:
                ACTIVE_TRAIN_PROC.kill()
            ACTIVE_TRAIN_PROC = None
            return "🛑 Training aborted by user.", "<div style='background: #fff7ed; color: #c2410c; padding: 6px 10px; border-radius: 6px; border: 1px solid #fed7aa; font-size: 0.85rem; font-weight: 600;'>🛑 Training process terminated by user.</div>"
        except Exception as err:
            return f"Error stopping training: {err}", f"<div style='color: #dc2626;'>{err}</div>"
    return "No active training process found.", "<div style='color: #64748b; font-size: 0.85rem;'>Idle - No training process active.</div>"

def _is_pkg_installed(pkg_name):
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        return False

# ==============================================================================
# Tab 5: System Diagnostics Handler
# ==============================================================================
def get_diagnostics():
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
# Gradio UI Construction & Compact Layout
# ==============================================================================
custom_css = """
/* Full display height layout - zero outer scrolling */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 4px 12px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    overflow-y: auto !important;
}

/* Modern sleek slim scrollbars */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
* {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
}

/* Compact header banner */
.header-banner {
    padding: 6px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    flex-shrink: 0 !important;
}

/* Compact tabs bar */
.tabs {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}
.tabs > .tab-nav {
    margin-bottom: 4px !important;
    flex-shrink: 0 !important;
}
.tabs > .tab-nav > button {
    padding: 4px 12px !important;
    font-size: 0.84rem !important;
}
.tabitem {
    flex: 1 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
}

/* Scrollable column container for rich forms without full-page scrolling */
.col-scroll {
    max-height: calc(100vh - 120px) !important;
    overflow-y: auto !important;
    padding-right: 4px !important;
}

/* Compact Form Components */
.gr-button {
    min-height: 32px !important;
    padding: 4px 10px !important;
    font-size: 0.84rem !important;
}
.gr-input, .gr-select {
    padding: 2px 6px !important;
    font-size: 0.85rem !important;
}
.gr-form {
    gap: 4px !important;
}
.accordion {
    margin-bottom: 3px !important;
}
.accordion > .label-wrap {
    padding: 3px 8px !important;
    font-size: 0.84rem !important;
}

/* Responsive Bengali & English Typography */
.bangla-output textarea {
    font-size: 1.15rem !important;
    line-height: 1.6 !important;
    font-family: 'SolaimanLipi', 'Noto Sans Bengali', 'Hind Siliguri', 'Segoe UI', system-ui, sans-serif !important;
    color: #0f172a !important;
    background-color: #f8fafc !important;
    border-radius: 6px !important;
}

/* Console log styling */
.console-log textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.35 !important;
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    border-radius: 6px !important;
}

/* Pulsing live streaming indicator */
@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}
.pulsing-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background-color: #0284c7;
    border-radius: 50%;
    animation: pulse 1.5s infinite ease-in-out;
}
"""

theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Noto Sans Bengali", "sans-serif"]
)

with gr.Blocks(title="Bangla & English ASR - Whisper") as demo:
    # Header Banner (Compact)
    gr.HTML("""
    <div class="header-banner" style="background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%); color: white; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <h1 style="font-size: 1.15rem; font-weight: 700; margin: 0;">
                🎙️ Bangla & English ASR Studio
            </h1>
            <span style="font-size: 0.80rem; opacity: 0.9;">
                Whisper large-v3-turbo with CTranslate2 INT8 & LoRA Fine-Tuning
            </span>
        </div>
        <div style="display: flex; gap: 6px;">
            <span style="background: rgba(255,255,255,0.18); padding: 3px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600;">
                🇧🇩 বাংলা & 🇬🇧 English
            </span>
            <span style="background: rgba(255,255,255,0.18); padding: 3px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600;">
                ⚡ INT8 Inference
            </span>
            <span style="background: rgba(255,255,255,0.18); padding: 3px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600;">
                🚀 large-v3-turbo
            </span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ======================================================================
        # TAB 1: Live Microphone & Single File (Compact Dashboard)
        # ======================================================================
        with gr.TabItem("🎯 Live Mic & Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record Speech or Upload Audio File"
                    )

                    with gr.Accordion("⚙️ Model & Quality Tuning", open=False):
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=["large-v3-turbo", "tiny"],
                                value="large-v3-turbo",
                                label="Checkpoint",
                                scale=1
                            )
                            lang_dropdown = gr.Dropdown(
                                choices=[
                                    "Auto (Smart Bilingual: Bangla / English)",
                                    "Bangla (বাংলা)",
                                    "English (en)"
                                ],
                                value="Auto (Smart Bilingual: Bangla / English)",
                                label="Language",
                                scale=1
                            )

                        with gr.Row():
                            beam_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Size")
                            temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")

                        vad_checkbox = gr.Checkbox(value=True, label="Enable VAD (Silence Trimming)")
                        prompt_input = gr.Textbox(
                            label="Context Hints / Vocabulary",
                            placeholder="e.g. বাংলাদেশ, কৃত্রিম বুদ্ধিমত্তা, ঢাকা",
                            lines=1
                        )

                    with gr.Row():
                        transcribe_btn = gr.Button("🚀 Transcribe", variant="primary", size="sm", scale=2)
                        stop_btn = gr.Button("🛑 Stop", variant="stop", size="sm", scale=1)
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm", scale=1)

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

                with gr.Column(scale=1):
                    with gr.Row():
                        gr.Markdown("##### 📝 Transcribed Output")
                        copy_btn = gr.Button("📋 Copy Text", size="sm", variant="secondary")

                    output_text = gr.Textbox(
                        label="",
                        show_label=False,
                        lines=5,
                        placeholder="Live streaming output will appear here as decoded...",
                        elem_classes=["bangla-output"]
                    )

                    metrics_output = gr.HTML()

                    with gr.Row():
                        download_txt = gr.DownloadButton("📄 Text", visible=False, size="sm")
                        download_srt = gr.DownloadButton("🎬 Subtitles (.srt)", visible=False, size="sm")
                        download_vtt = gr.DownloadButton("🌐 WebVTT (.vtt)", visible=False, size="sm")
                        download_json = gr.DownloadButton("📦 JSON", visible=False, size="sm")

                    with gr.Accordion("⏱️ Segments Breakdown", open=False):
                        segments_table = gr.DataFrame(
                            headers=["Start (s)", "End (s)", "Duration (s)", "Transcription"],
                            label="Speech Segments",
                            wrap=True,
                            max_height=200
                        )

                    with gr.Accordion("🔍 Full JSON Metadata", open=False):
                        json_output = gr.Code(language="json", label="Raw JSON Output")

            transcribe_event = transcribe_btn.click(
                fn=transcribe_audio_streaming,
                inputs=[audio_input, model_dropdown, lang_dropdown, beam_slider, temp_slider, prompt_input, vad_checkbox],
                outputs=[output_text, metrics_output, segments_table, json_output, download_txt, download_srt, download_vtt, download_json]
            )

            stop_btn.click(
                fn=on_single_transcribe_stop,
                inputs=None,
                outputs=[metrics_output],
                cancels=[transcribe_event]
            )

            copy_btn.click(
                fn=None,
                inputs=[output_text],
                js="(text) => { if (text) { navigator.clipboard.writeText(text); } }"
            )

            clear_btn.click(
                fn=on_single_clear,
                outputs=[audio_input, output_text, metrics_output, segments_table, json_output, download_txt, download_srt, download_vtt, download_json],
                cancels=[transcribe_event]
            )

        # ======================================================================
        # TAB 2: Batch Audio Transcription (Compact)
        # ======================================================================
        with gr.TabItem("📂 Batch Transcription"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_mode = gr.Radio(
                        choices=["Upload Files directly in Browser", "Specify Server Directory Path"],
                        value="Upload Files directly in Browser",
                        label="Source Mode"
                    )

                    batch_files_upload = gr.File(
                        file_count="multiple",
                        file_types=["audio"],
                        label="Drop Audio Files Here",
                        visible=True
                    )

                    batch_dir_input = gr.Textbox(
                        value="data/test/audio",
                        label="Server Directory Path",
                        visible=False
                    )

                    batch_mode.change(
                        fn=lambda mode: (gr.update(visible=mode == "Upload Files directly in Browser"), gr.update(visible=mode != "Upload Files directly in Browser")),
                        inputs=[batch_mode],
                        outputs=[batch_files_upload, batch_dir_input]
                    )

                    with gr.Row():
                        batch_model = gr.Dropdown(choices=["large-v3-turbo", "tiny"], value="large-v3-turbo", label="Model")
                        batch_lang = gr.Dropdown(
                            choices=["Auto (Smart Bilingual: Bangla / English)", "Bangla (বাংলা)", "English (en)"],
                            value="Auto (Smart Bilingual: Bangla / English)",
                            label="Language"
                        )

                    with gr.Row():
                        batch_btn = gr.Button("⚡ Start Batch", variant="primary", size="sm", scale=2)
                        batch_stop_btn = gr.Button("🛑 Stop Batch", variant="stop", size="sm", scale=1)

                with gr.Column(scale=1):
                    batch_status_html = gr.HTML()
                    with gr.Row():
                        batch_download_csv = gr.DownloadButton("📥 Download CSV", visible=False, size="sm")
                        batch_download_json = gr.DownloadButton("📥 Download JSON", visible=False, size="sm")

                    batch_table = gr.DataFrame(
                        headers=["File", "Language", "Confidence", "Duration (s)", "Speed", "Transcription"],
                        label="Batch Results (updates live)",
                        wrap=True,
                        max_height=320
                    )

            batch_event = batch_btn.click(
                fn=batch_transcribe_streaming,
                inputs=[batch_mode, batch_files_upload, batch_dir_input, batch_model, batch_lang],
                outputs=[batch_status_html, batch_table, batch_download_csv, batch_download_json]
            )

            batch_stop_btn.click(
                fn=on_batch_stop,
                outputs=[batch_status_html],
                cancels=[batch_event]
            )

        # ======================================================================
        # TAB 3: Dataset Benchmark (WER & CER) (Compact)
        # ======================================================================
        with gr.TabItem("📊 Benchmark (WER / CER)"):
            with gr.Row():
                with gr.Column(scale=1):
                    eval_csv_upload = gr.File(label="Upload CSV (Optional)", file_types=[".csv"])
                    with gr.Row():
                        eval_csv_input = gr.Textbox(value="data/test/metadata.csv", label="Metadata CSV Path")
                        eval_audio_dir = gr.Textbox(value="data/test/audio", label="Audio Directory")
                    with gr.Row():
                        eval_model = gr.Dropdown(choices=["large-v3-turbo", "tiny"], value="large-v3-turbo", label="Model")
                        eval_lang = gr.Dropdown(
                            choices=["Auto (Smart Bilingual: Bangla / English)", "Bangla (বাংলা)", "English (en)"],
                            value="Auto (Smart Bilingual: Bangla / English)",
                            label="Language"
                        )

                    with gr.Row():
                        eval_btn = gr.Button("📈 Run Benchmark", variant="primary", size="sm", scale=2)
                        eval_stop_btn = gr.Button("🛑 Stop", variant="stop", size="sm", scale=1)

                with gr.Column(scale=1):
                    eval_status = gr.Markdown()
                    eval_metrics_html = gr.HTML()
                    eval_download_csv = gr.DownloadButton("📥 Download Error CSV", visible=False, size="sm")

                    eval_results_table = gr.DataFrame(
                        headers=["File", "Ground Truth", "Prediction", "WER", "CER"],
                        label="Predictions vs Ground Truth",
                        wrap=True,
                        max_height=320
                    )

            eval_event = eval_btn.click(
                fn=evaluate_dataset_streaming,
                inputs=[eval_csv_upload, eval_csv_input, eval_audio_dir, eval_model, eval_lang],
                outputs=[eval_status, eval_metrics_html, eval_results_table, eval_download_csv]
            )

            eval_stop_btn.click(
                fn=on_eval_stop,
                outputs=[eval_status],
                cancels=[eval_event]
            )

        # ======================================================================
        # TAB 4: Batched Training Studio (All 36 Parameters + Adaptive Layout)
        # ======================================================================
        with gr.TabItem("🏋️ Batched Training"):
            with gr.Row():
                # Left Column: All Hyperparameters organized compactly
                with gr.Column(scale=1, elem_classes=["col-scroll"]):
                    with gr.Accordion("🔍 Hardware & Dependencies Check", open=False):
                        train_env_markdown = gr.Markdown(value=check_training_environment())
                        check_env_btn = gr.Button("🔄 Re-Check GPU", size="sm", variant="secondary")
                        check_env_btn.click(fn=check_training_environment, outputs=[train_env_markdown])

                    # 1. Base Model & Architecture
                    with gr.Accordion("🤖 1. Base Model Checkpoint (Auto-Adaptive)", open=True):
                        base_model_dropdown = gr.Dropdown(
                            choices=[
                                "openai/whisper-large-v3-turbo",
                                "openai/whisper-large-v3",
                                "openai/whisper-medium",
                                "openai/whisper-small",
                                "openai/whisper-base",
                                "openai/whisper-tiny"
                            ],
                            value="openai/whisper-large-v3-turbo",
                            label="Base Model (select to auto-adapt parameters)"
                        )
                        model_desc_markdown = gr.Markdown(value=MODEL_PRESETS["openai/whisper-large-v3-turbo"]["description"])

                        with gr.Row():
                            target_lang_dropdown = gr.Dropdown(choices=["bengali", "english"], value="bengali", label="Target Language")
                            task_dropdown = gr.Dropdown(choices=["transcribe", "translate"], value="transcribe", label="Task")

                    # 2. Datasets & Preprocessing
                    with gr.Accordion("📁 2. Dataset Paths & Audio Filtering", open=True):
                        with gr.Row():
                            train_csv_box = gr.Textbox(value="data/train/metadata.csv", label="Train CSV")
                            train_audio_box = gr.Textbox(value="data/train/audio", label="Train Audio Folder")
                        with gr.Row():
                            val_csv_box = gr.Textbox(value="data/val/metadata.csv", label="Val CSV")
                            val_audio_box = gr.Textbox(value="data/val/audio", label="Val Audio Folder")

                        with gr.Row():
                            num_proc_slider = gr.Slider(minimum=1, maximum=8, value=2, step=1, label="Mel CPU Workers (num_proc)")
                            dataloader_workers_slider = gr.Slider(minimum=0, maximum=8, value=2, step=1, label="DataLoader Workers")

                        verify_data_btn = gr.Button("🔍 Verify Dataset Integrity", size="sm", variant="secondary")
                        data_verify_output = gr.Markdown()
                        verify_data_btn.click(
                            fn=validate_training_dataset_gui,
                            inputs=[train_csv_box, train_audio_box, val_csv_box, val_audio_box],
                            outputs=[data_verify_output]
                        )

                    # 3. Batching & Compute
                    with gr.Accordion("⚡ 3. Batching & Memory Optimization", open=True):
                        with gr.Row():
                            train_batch_slider = gr.Slider(minimum=1, maximum=64, value=8, step=1, label="Train Batch Size")
                            eval_batch_slider = gr.Slider(minimum=1, maximum=64, value=8, step=1, label="Eval Batch Size")
                            grad_accum_slider = gr.Slider(minimum=1, maximum=32, value=2, step=1, label="Grad Accum Steps")

                        batch_info_display = gr.Markdown("💡 **Effective Batch Size**: `8 × 2 = 16 samples per step`")

                        def update_effective_batch(b, g):
                            return f"💡 **Effective Batch Size**: `{int(b)} × {int(g)} = {int(b) * int(g)} samples per step`"

                        train_batch_slider.change(fn=update_effective_batch, inputs=[train_batch_slider, grad_accum_slider], outputs=[batch_info_display])
                        grad_accum_slider.change(fn=update_effective_batch, inputs=[train_batch_slider, grad_accum_slider], outputs=[batch_info_display])

                        with gr.Row():
                            precision_radio = gr.Radio(
                                choices=["FP16 Mixed Precision", "BF16 (Ampere/Ada)", "FP32 (Standard)"],
                                value="FP16 Mixed Precision",
                                label="Precision"
                            )
                            grad_ckpt_check = gr.Checkbox(value=True, label="Gradient Checkpointing")

                    # 4. LoRA / QLoRA
                    with gr.Accordion("🧩 4. LoRA / QLoRA Architecture", open=False):
                        finetune_mode_radio = gr.Radio(
                            choices=[
                                "LoRA (Parameter-Efficient PEFT) - Recommended",
                                "QLoRA (4-Bit NF4 Quantization)",
                                "Full Model Fine-Tuning"
                            ],
                            value="LoRA (Parameter-Efficient PEFT) - Recommended",
                            label="Fine-Tuning Mode"
                        )

                        with gr.Row():
                            lora_r_slider = gr.Slider(minimum=4, maximum=128, value=32, step=4, label="LoRA Rank (r)")
                            lora_alpha_slider = gr.Slider(minimum=8, maximum=256, value=64, step=8, label="LoRA Alpha")
                            lora_dropout_slider = gr.Slider(minimum=0.0, maximum=0.2, value=0.05, step=0.01, label="LoRA Dropout")

                        lora_target_box = gr.Textbox(
                            value="q_proj,v_proj",
                            label="LoRA Attention Modules (comma-separated)",
                            placeholder="q_proj,v_proj or q_proj,k_proj,v_proj,out_proj,fc1,fc2"
                        )

                    # 5. Optimizer & Scheduler
                    with gr.Accordion("🎯 5. Optimizer, Learning Rate & Scheduler", open=False):
                        with gr.Row():
                            optim_dropdown = gr.Dropdown(choices=["adamw_torch", "adamw_bnb_8bit", "adafactor", "sgd"], value="adamw_torch", label="Optimizer")
                            lr_input = gr.Dropdown(choices=["1e-5", "3e-5", "5e-5", "1e-4", "2e-4", "5e-4"], value="1e-4", label="Learning Rate")
                            scheduler_dropdown = gr.Dropdown(choices=["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"], value="linear", label="Scheduler")

                        with gr.Row():
                            warmup_slider = gr.Slider(minimum=0, maximum=500, value=50, step=10, label="Warmup Steps")
                            weight_decay_slider = gr.Slider(minimum=0.0, maximum=0.2, value=0.01, step=0.005, label="Weight Decay")
                            max_grad_norm_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Max Grad Norm")

                    # 6. Duration & Checkpointing
                    with gr.Accordion("⏱️ 6. Duration, Steps & Checkpointing", open=False):
                        with gr.Row():
                            epochs_slider = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Total Epochs")
                            max_steps_box = gr.Number(value=-1, label="Max Steps (-1 for full epochs)")

                        with gr.Row():
                            eval_steps_box = gr.Number(value=200, label="Eval Steps")
                            save_steps_box = gr.Number(value=200, label="Save Steps")
                            logging_steps_box = gr.Number(value=25, label="Logging Steps")
                            save_limit_box = gr.Number(value=2, label="Max Checkpoints")

                        with gr.Row():
                            best_metric_dropdown = gr.Dropdown(choices=["wer", "cer", "loss"], value="wer", label="Best Model Metric")
                            report_to_dropdown = gr.Dropdown(choices=["tensorboard", "none", "wandb"], value="tensorboard", label="Logger")

                        output_dir_box = gr.Textbox(value="./checkpoints/whisper_bangla_lora", label="Checkpoint Output Directory")

                    # 7. Evaluation Generation
                    with gr.Accordion("🎙️ 7. Evaluation Generation Decoding", open=False):
                        with gr.Row():
                            gen_len_slider = gr.Slider(minimum=64, maximum=448, value=225, step=1, label="Generation Max Tokens")
                            gen_beams_slider = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Generation Beams")

                    # Auto-adaptation of hyperparameters on model change
                    base_model_dropdown.change(
                        fn=apply_model_preset,
                        inputs=[base_model_dropdown],
                        outputs=[
                            train_batch_slider,
                            eval_batch_slider,
                            grad_accum_slider,
                            lr_input,
                            lora_r_slider,
                            lora_alpha_slider,
                            lora_target_box,
                            finetune_mode_radio,
                            precision_radio,
                            optim_dropdown,
                            model_desc_markdown,
                            batch_info_display
                        ]
                    )

                    with gr.Row():
                        train_btn = gr.Button("🚀 Launch Batched Training", variant="primary", size="sm", scale=2)
                        train_stop_btn = gr.Button("🛑 Abort Training", variant="stop", size="sm", scale=1)

                    with gr.Accordion("📋 View CLI Command", open=False):
                        cli_code_output = gr.Code(language="shell", label="Command for Remote Servers / Cloud")
                        show_cmd_btn = gr.Button("Generate Command", size="sm", variant="secondary")

                # Right Column: Live Terminal & Training Console
                with gr.Column(scale=1):
                    train_status_banner = gr.HTML(value="<div style='color: #64748b; font-size: 0.85rem;'>Ready to train. Press <b>Launch Batched Training</b>.</div>")
                    train_log_box = gr.Textbox(
                        label="🖥️ Live Console & Loss Output",
                        lines=16,
                        placeholder="Training output, step loss, evaluation WER/CER, and checkpoints will stream here...",
                        elem_classes=["console-log"]
                    )

            all_train_inputs = [
                base_model_dropdown, target_lang_dropdown, task_dropdown,
                train_csv_box, train_audio_box, val_csv_box, val_audio_box, output_dir_box, num_proc_slider,
                finetune_mode_radio, lora_r_slider, lora_alpha_slider, lora_dropout_slider, lora_target_box,
                train_batch_slider, eval_batch_slider, grad_accum_slider, grad_ckpt_check, precision_radio, dataloader_workers_slider,
                optim_dropdown, lr_input, scheduler_dropdown, warmup_slider, weight_decay_slider, max_grad_norm_slider,
                epochs_slider, max_steps_box, eval_steps_box, save_steps_box, logging_steps_box, save_limit_box, best_metric_dropdown, report_to_dropdown,
                gen_len_slider, gen_beams_slider
            ]

            show_cmd_btn.click(
                fn=build_cli_command,
                inputs=all_train_inputs,
                outputs=[cli_code_output]
            )

            train_btn.click(
                fn=start_training_gui,
                inputs=all_train_inputs,
                outputs=[train_log_box, train_status_banner]
            )

            train_stop_btn.click(
                fn=stop_training_gui,
                outputs=[train_log_box, train_status_banner]
            )

        # ======================================================================
        # TAB 5: System Diagnostics & Health Check (Compact)
        # ======================================================================
        with gr.TabItem("🖥️ Diagnostics"):
            diag_output = gr.Markdown(value=get_diagnostics())
            refresh_diag_btn = gr.Button("🔄 Refresh Diagnostics", size="sm", variant="secondary")
            refresh_diag_btn.click(fn=get_diagnostics, outputs=[diag_output])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=custom_css)
