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
# Tab 1: Streaming Single Audio & Microphone Handler with Stop Support
# ==============================================================================
def transcribe_audio_streaming(audio_path, model_name, language_choice, beam_size, temperature, initial_prompt, vad_filter):
    if not audio_path:
        yield (
            "Please record speech using your microphone or upload an audio file (.wav, .mp3, .flac).",
            "<div style='color: #64748b; font-size: 0.9rem;'>Waiting for audio input...</div>",
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
        "<div style='background: #eff6ff; color: #1d4ed8; padding: 10px 14px; border-radius: 8px; border: 1px solid #bfdbfe; font-weight: 600;'>⏳ Loading Whisper model & initializing audio stream...</div>",
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
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #0284c7; display: flex; align-items: center; gap: 6px;">
                        <span class="pulsing-dot"></span> ⏳ Transcribing live ({seg.end:.1f}s / {total_duration:.1f}s)...
                    </span>
                    <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">
                        {flag} {detected_lang} ({lang_prob:.1%})
                    </span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 6px; overflow: hidden;">
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
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; margin-bottom: 8px;">
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 140px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Language Detected</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                    {flag} {detected_lang} <span style="font-size: 0.85rem; font-weight: 400; color: #059669;">({lang_prob:.1%})</span>
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 130px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Audio Duration</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                    ⏱️ {total_duration:.2f}s
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 140px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Processing Speed</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #059669; margin-top: 2px;">
                    🚀 {final_speed}x real-time <span style="font-size: 0.8rem; font-weight: 400; color: #64748b;">({elapsed_total:.2f}s)</span>
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; flex: 1; min-width: 120px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Word Count</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                    📊 {word_count} words <span style="font-size: 0.8rem; font-weight: 400; color: #64748b;">({char_count} chars)</span>
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
            f"<div style='color: #dc2626; font-weight: 600; padding: 8px 12px; background: #fee2e2; border-radius: 6px;'>❌ {error_msg}</div>",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def on_single_transcribe_stop():
    return "<div style='color: #c2410c; font-weight: 600; padding: 10px 14px; background: #fff7ed; border-radius: 8px; border: 1px solid #fed7aa;'>🛑 Transcription stopped by user. You can modify audio, settings, or restart anytime.</div>"

def on_single_clear():
    return (
        None,
        "",
        "<div style='color: #64748b; font-size: 0.9rem;'>All inputs cleared. Ready for new audio.</div>",
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
        f"<div style='background: #eff6ff; color: #1d4ed8; padding: 10px 14px; border-radius: 8px; border: 1px solid #bfdbfe; font-weight: 600;'>⏳ Initializing batch processing for {len(audio_paths)} files...</div>",
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
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">
            <div style="background: #ecfdf5; border: 1px solid #6ee7b7; padding: 12px 16px; border-radius: 8px; flex: 1;">
                <span style="color: #065f46; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Progress</span>
                <div style="font-size: 1.25rem; font-weight: 700; color: #064e3b; margin-top: 2px;">
                    ✅ {idx + 1} / {len(audio_paths)} Files
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

        yield (
            running_html,
            df_curr,
            gr.update(value=out_csv, visible=True),
            gr.update(value=out_json, visible=True)
        )

def on_batch_stop():
    return "<div style='color: #c2410c; font-weight: 600; padding: 10px 14px; background: #fff7ed; border-radius: 8px; border: 1px solid #fed7aa;'>🛑 Batch processing stopped by user. Files completed so far are displayed below and ready to download.</div>"

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
        f"<div style='background: #eff6ff; color: #1d4ed8; padding: 10px 14px; border-radius: 8px; border: 1px solid #bfdbfe; font-weight: 600;'>⏳ Initializing model and preparing {len(df)} dataset rows...</div>",
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
        <div style="display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 12px;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Evaluated Samples</span>
                <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 4px;">
                    📈 {len(results_df)} / {len(df)}
                </div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Running WER</span>
                <div style="font-size: 1.5rem; font-weight: 700; color: {wer_color}; margin-top: 4px;">
                    🎯 {running_wer:.2%}
                </div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Running CER</span>
                <div style="font-size: 1.5rem; font-weight: 700; color: {cer_color}; margin-top: 4px;">
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
    return "<div style='color: #c2410c; font-weight: 600; padding: 10px 14px; background: #fff7ed; border-radius: 8px; border: 1px solid #fed7aa;'>🛑 Benchmark stopped by user. Partial evaluation results and metrics are preserved below.</div>"

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
            checks.append(f"🟢 **Compute Hardware**: NVIDIA CUDA GPU `{dev}` ({mem:.1f} GB VRAM) is ready.")
        else:
            checks.append("🟡 **Compute Hardware**: No CUDA GPU found (running in CPU mode). CPU can be used to test small mini-batches, but fine-tuning `large-v3-turbo` on full datasets requires a GPU (≥16GB VRAM recommended).")
    except Exception as e:
        checks.append(f"🔴 **Compute Hardware**: {e}")

    missing = []
    for pkg in ["transformers", "datasets", "peft", "accelerate", "evaluate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        checks.append(f"⚠️ **Missing Training Packages**: `{', '.join(missing)}`\n> **To install**: Run `pip install -r requirements_gpu.txt` in your terminal.")
    else:
        checks.append("🟢 **All Training Packages Installed**: `transformers`, `datasets`, `peft`, `accelerate`, and `evaluate` are installed.")

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
    """Adapts all training hyperparameters to optimal defaults when base model changes."""
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
        error_txt = f"❌ Missing required packages for training: {', '.join(missing)}\nPlease install training requirements:\n  pip install -r requirements_gpu.txt\n"
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

    status_html = f"<div style='background: #eff6ff; color: #1e40af; padding: 10px 14px; border-radius: 8px; border: 1px solid #93c5fd; font-weight: 600;'>⏳ Training running (Effective Batch Size: {effective_batch})...</div>"
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
            if len(log_lines) > 400:
                log_lines = log_lines[-400:]
            yield "\n".join(log_lines), status_html

        ACTIVE_TRAIN_PROC.stdout.close()
        code = ACTIVE_TRAIN_PROC.wait()
        ACTIVE_TRAIN_PROC = None

        if code == 0:
            log_lines.append(f"\n🎉 TRAINING FINISHED SUCCESSFULLY! Checkpoints stored in: {output_dir}")
            final_status = "<div style='background: #ecfdf5; color: #065f46; padding: 10px 14px; border-radius: 8px; border: 1px solid #6ee7b7; font-weight: 600;'>✅ Training completed successfully!</div>"
        else:
            log_lines.append(f"\n⚠️ Process exited with return code: {code}")
            final_status = f"<div style='background: #fff1f2; color: #9f1239; padding: 10px 14px; border-radius: 8px; border: 1px solid #fecdd3; font-weight: 600;'>⚠️ Training stopped or failed (exit code {code}).</div>"

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
            return "🛑 Training aborted by user.", "<div style='background: #fff7ed; color: #c2410c; padding: 10px 14px; border-radius: 8px; border: 1px solid #fed7aa; font-weight: 600;'>🛑 Training process terminated by user.</div>"
        except Exception as err:
            return f"Error stopping training: {err}", f"<div style='color: #dc2626;'>{err}</div>"
    return "No active training process found.", "<div style='color: #64748b;'>Idle - No training process active.</div>"

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
# Gradio UI Construction & Styling
# ==============================================================================
custom_css = """
/* Responsive Bengali & English Typography */
.bangla-output textarea {
    font-size: 1.25rem !important;
    line-height: 1.85 !important;
    font-family: 'SolaimanLipi', 'Noto Sans Bengali', 'Hind Siliguri', 'Segoe UI', system-ui, sans-serif !important;
    color: #0f172a !important;
    background-color: #f8fafc !important;
    border-radius: 8px !important;
}

/* Console log styling */
.console-log textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    border-radius: 8px !important;
}

/* Pulsing live streaming indicator */
@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}
.pulsing-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
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
    # Header Banner
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%); padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 style="font-size: 1.75rem; font-weight: 800; margin: 0; display: flex; align-items: center; gap: 10px;">
                    🎙️ Bangla & English Speech Recognition (ASR)
                </h1>
                <p style="font-size: 0.95rem; margin: 6px 0 0 0; opacity: 0.9;">
                    High-accuracy speech-to-text with live streaming chunks, batched audio training from GUI, and smart bilingual language routing.
                </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; backdrop-filter: blur(4px);">
                    🇧🇩 বাংলা & 🇬🇧 English
                </span>
                <span style="background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; backdrop-filter: blur(4px);">
                    ⚡ INT8 Inference & Full Parameter Training
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
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record Speech or Upload Audio File (.wav, .mp3, .flac, .ogg, .m4a)"
                    )

                    with gr.Accordion("⚙️ Model, Decoding & Quality Settings", open=True):
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
                                label="Beam Size (Accuracy vs Latency)"
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
                            label="Context Hints / Initial Vocabulary (Optional)",
                            placeholder="e.g. বাংলাদেশ, কৃত্রিম বুদ্ধিমত্তা, ঢাকা, Whisper ASR",
                            lines=1
                        )

                    with gr.Row():
                        transcribe_btn = gr.Button("🚀 Transcribe Speech", variant="primary", size="lg", scale=2)
                        stop_btn = gr.Button("🛑 Stop / Cancel", variant="stop", size="lg", scale=1)
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", scale=1)

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
                        gr.Markdown("#### 📝 Transcribed Output")
                        copy_btn = gr.Button("📋 Copy Text", size="sm", variant="secondary")

                    output_text = gr.Textbox(
                        label="",
                        show_label=False,
                        lines=7,
                        placeholder="Transcribed speech will appear here in real-time as it's being decoded...",
                        elem_classes=["bangla-output"]
                    )

                    metrics_output = gr.HTML()

                    gr.Markdown("##### 💾 Download & Subtitle Exports")
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
        # TAB 2: Batch Audio Transcription with Stop Support
        # ======================================================================
        with gr.TabItem("📂 Batch Audio Transcription"):
            gr.Markdown("""
            ### Batch Transcribe Audio Files
            Process multiple audio files. Drag and drop audio files directly in the browser, or specify a server directory.
            You can stop batch processing at any point; completed files will be preserved.
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

                    with gr.Row():
                        batch_btn = gr.Button("⚡ Start Batch Transcription", variant="primary", size="lg", scale=2)
                        batch_stop_btn = gr.Button("🛑 Stop Batch", variant="stop", size="lg", scale=1)

                with gr.Column(scale=1):
                    batch_status_html = gr.HTML()
                    with gr.Row():
                        batch_download_csv = gr.DownloadButton("📥 Download CSV Report", visible=False)
                        batch_download_json = gr.DownloadButton("📥 Download JSON Results", visible=False)

            batch_table = gr.DataFrame(
                headers=["File", "Language", "Confidence", "Duration (s)", "Speed", "Transcription"],
                label="Batch Results (updates live)",
                wrap=True
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
        # TAB 3: Dataset Benchmark (WER & CER) with Stop Support
        # ======================================================================
        with gr.TabItem("📊 Benchmark & Evaluate (WER / CER)"):
            gr.Markdown("""
            ### Accuracy Benchmark Tool
            Evaluate speech recognition accuracy using **Word Error Rate (WER)** and **Character Error Rate (CER)** against a test dataset with ground truth references.
            You can stop benchmarking at any time without losing evaluated sample metrics.
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

                    with gr.Row():
                        eval_btn = gr.Button("📈 Run Benchmark", variant="primary", size="lg", scale=2)
                        eval_stop_btn = gr.Button("🛑 Stop Evaluation", variant="stop", size="lg", scale=1)

                with gr.Column(scale=1):
                    eval_status = gr.Markdown()
                    eval_metrics_html = gr.HTML()
                    eval_download_csv = gr.DownloadButton("📥 Download Error Report (CSV)", visible=False)

            eval_results_table = gr.DataFrame(
                headers=["File", "Ground Truth", "Prediction", "WER", "CER"],
                label="Sample Predictions vs Ground Truth (updates live)",
                wrap=True
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
        # TAB 4: Batched Audio Training / Fine-Tuning Manager (Full Params)
        # ======================================================================
        with gr.TabItem("🏋️ Train / Fine-Tune (Batched)"):
            gr.Markdown("""
            ### 🏋️ Whisper Batched Audio Training & Fine-Tuning Studio
            Fine-tune Whisper models on customized Bengali and English speech datasets with **complete parameter control**.
            Changing the base model automatically adapts optimal batching, learning rate, and architecture presets.
            """)

            with gr.Accordion("🔍 Hardware Readiness & Dependencies Check", open=False):
                train_env_markdown = gr.Markdown(value=check_training_environment())
                check_env_btn = gr.Button("🔄 Re-Check GPU & Packages", size="sm", variant="secondary")
                check_env_btn.click(fn=check_training_environment, outputs=[train_env_markdown])

            with gr.Row():
                # Left Column: Complete Parameter Controls
                with gr.Column(scale=1):
                    # Section 1: Model & Architecture Selection
                    with gr.Accordion("🤖 1. Base Model & Architecture (Auto-Adaptive)", open=True):
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
                            label="Whisper Base Checkpoint (select to auto-adapt parameters)"
                        )
                        model_desc_markdown = gr.Markdown(value=MODEL_PRESETS["openai/whisper-large-v3-turbo"]["description"])

                        with gr.Row():
                            target_lang_dropdown = gr.Dropdown(
                                choices=["bengali", "english"],
                                value="bengali",
                                label="Target Language"
                            )
                            task_dropdown = gr.Dropdown(
                                choices=["transcribe", "translate"],
                                value="transcribe",
                                label="Task"
                            )

                    # Section 2: Datasets & Preprocessing
                    with gr.Accordion("📁 2. Dataset Paths & Audio Filtering", open=True):
                        with gr.Row():
                            train_csv_box = gr.Textbox(value="data/train/metadata.csv", label="Train Metadata CSV Path")
                            train_audio_box = gr.Textbox(value="data/train/audio", label="Train Audio Directory")
                        with gr.Row():
                            val_csv_box = gr.Textbox(value="data/val/metadata.csv", label="Validation Metadata CSV Path")
                            val_audio_box = gr.Textbox(value="data/val/audio", label="Validation Audio Directory")

                        with gr.Row():
                            num_proc_slider = gr.Slider(minimum=1, maximum=8, value=2, step=1, label="CPU Worker Processes (num_proc)")
                            dataloader_workers_slider = gr.Slider(minimum=0, maximum=8, value=2, step=1, label="DataLoader Workers")

                        verify_data_btn = gr.Button("🔍 Verify Dataset Files & Audio Integrity", size="sm", variant="secondary")
                        data_verify_output = gr.Markdown()
                        verify_data_btn.click(
                            fn=validate_training_dataset_gui,
                            inputs=[train_csv_box, train_audio_box, val_csv_box, val_audio_box],
                            outputs=[data_verify_output]
                        )

                    # Section 3: Batching & Memory Optimization
                    with gr.Accordion("⚡ 3. Batching & Memory Optimization", open=True):
                        with gr.Row():
                            train_batch_slider = gr.Slider(minimum=1, maximum=64, value=8, step=1, label="Per-Device Train Batch Size")
                            eval_batch_slider = gr.Slider(minimum=1, maximum=64, value=8, step=1, label="Per-Device Eval Batch Size")
                            grad_accum_slider = gr.Slider(minimum=1, maximum=32, value=2, step=1, label="Gradient Accumulation Steps")

                        batch_info_display = gr.Markdown("💡 **Effective Batch Size**: `8 × 2 = 16 samples per step`")

                        def update_effective_batch(b, g):
                            return f"💡 **Effective Batch Size**: `{int(b)} × {int(g)} = {int(b) * int(g)} samples per step`"

                        train_batch_slider.change(fn=update_effective_batch, inputs=[train_batch_slider, grad_accum_slider], outputs=[batch_info_display])
                        grad_accum_slider.change(fn=update_effective_batch, inputs=[train_batch_slider, grad_accum_slider], outputs=[batch_info_display])

                        with gr.Row():
                            precision_radio = gr.Radio(
                                choices=["FP16 Mixed Precision", "BF16 (Ampere/Ada)", "FP32 (Standard)"],
                                value="FP16 Mixed Precision",
                                label="Compute Precision"
                            )
                            grad_ckpt_check = gr.Checkbox(value=True, label="Enable Gradient Checkpointing (Saves VRAM)")

                    # Section 4: Parameter-Efficient Fine-Tuning (PEFT / LoRA / QLoRA)
                    with gr.Accordion("🧩 4. LoRA / QLoRA & Quantization Parameters", open=True):
                        finetune_mode_radio = gr.Radio(
                            choices=[
                                "LoRA (Parameter-Efficient PEFT) - Recommended",
                                "QLoRA (4-Bit NF4 Quantization)",
                                "Full Model Fine-Tuning"
                            ],
                            value="LoRA (Parameter-Efficient PEFT) - Recommended",
                            label="Fine-Tuning Architecture Mode"
                        )

                        with gr.Row():
                            lora_r_slider = gr.Slider(minimum=4, maximum=128, value=32, step=4, label="LoRA Rank (r)")
                            lora_alpha_slider = gr.Slider(minimum=8, maximum=256, value=64, step=8, label="LoRA Alpha (scaling)")
                            lora_dropout_slider = gr.Slider(minimum=0.0, maximum=0.2, value=0.05, step=0.01, label="LoRA Dropout")

                        lora_target_box = gr.Textbox(
                            value="q_proj,v_proj",
                            label="LoRA Target Attention Modules (comma-separated)",
                            placeholder="q_proj,v_proj or q_proj,k_proj,v_proj,out_proj,fc1,fc2"
                        )

                    # Section 5: Optimizer & Learning Rate Schedule
                    with gr.Accordion("🎯 5. Optimizer, Learning Rate & Scheduler", open=False):
                        with gr.Row():
                            optim_dropdown = gr.Dropdown(
                                choices=["adamw_torch", "adamw_bnb_8bit", "adafactor", "sgd"],
                                value="adamw_torch",
                                label="Optimizer"
                            )
                            lr_input = gr.Dropdown(
                                choices=["1e-5", "3e-5", "5e-5", "1e-4", "2e-4", "5e-4"],
                                value="1e-4",
                                label="Learning Rate"
                            )
                            scheduler_dropdown = gr.Dropdown(
                                choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"],
                                value="linear",
                                label="LR Scheduler Type"
                            )

                        with gr.Row():
                            warmup_slider = gr.Slider(minimum=0, maximum=500, value=50, step=10, label="Warmup Steps")
                            weight_decay_slider = gr.Slider(minimum=0.0, maximum=0.2, value=0.01, step=0.005, label="Weight Decay")
                            max_grad_norm_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Max Gradient Norm (Clipping)")

                    # Section 6: Training Duration & Checkpointing Schedule
                    with gr.Accordion("⏱️ 6. Training Duration, Steps & Checkpointing", open=False):
                        with gr.Row():
                            epochs_slider = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Total Epochs")
                            max_steps_box = gr.Number(value=-1, label="Max Steps (-1 for full epochs, >0 overrides epochs)")

                        with gr.Row():
                            eval_steps_box = gr.Number(value=200, label="Evaluation Frequency (steps)")
                            save_steps_box = gr.Number(value=200, label="Checkpoint Save Frequency (steps)")
                            logging_steps_box = gr.Number(value=25, label="Logging Steps")
                            save_limit_box = gr.Number(value=2, label="Max Checkpoints to Keep")

                        with gr.Row():
                            best_metric_dropdown = gr.Dropdown(choices=["wer", "cer", "loss"], value="wer", label="Metric for Best Model Selection")
                            report_to_dropdown = gr.Dropdown(choices=["tensorboard", "none", "wandb"], value="tensorboard", label="Dashboard Logger")

                        output_dir_box = gr.Textbox(value="./checkpoints/whisper_bangla_lora", label="Output Checkpoint Save Directory")

                    # Section 7: Evaluation Generation & Decoding
                    with gr.Accordion("🎙️ 7. Evaluation Generation & Decoding Parameters", open=False):
                        with gr.Row():
                            gen_len_slider = gr.Slider(minimum=64, maximum=448, value=225, step=1, label="Generation Max Length (tokens)")
                            gen_beams_slider = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Generation Num Beams (1 for fast eval)")

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
                        train_btn = gr.Button("🚀 Launch Batched Training", variant="primary", size="lg", scale=2)
                        train_stop_btn = gr.Button("🛑 Abort Training", variant="stop", size="lg", scale=1)

                    with gr.Accordion("📋 View Complete Equivalent CLI Command", open=False):
                        cli_code_output = gr.Code(language="shell", label="Command for Remote Servers / Cloud Clusters")
                        show_cmd_btn = gr.Button("Generate Command", size="sm", variant="secondary")

                # Right Column: Live Terminal & Training Console
                with gr.Column(scale=1):
                    train_status_banner = gr.HTML(value="<div style='color: #64748b; font-size: 0.95rem;'>Ready to train. Configure parameters and press <b>Launch Batched Training</b>.</div>")
                    train_log_box = gr.Textbox(
                        label="🖥️ Live Training Console & Loss Output",
                        lines=28,
                        placeholder="Training output, step loss, evaluation WER/CER, and checkpoint notifications will stream here live...",
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
        # TAB 5: System Diagnostics & Health Check
        # ======================================================================
        with gr.TabItem("🖥️ System Diagnostics & Cache"):
            diag_output = gr.Markdown(value=get_diagnostics())
            refresh_diag_btn = gr.Button("🔄 Refresh Diagnostics", variant="secondary")
            refresh_diag_btn.click(fn=get_diagnostics, outputs=[diag_output])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=custom_css)
