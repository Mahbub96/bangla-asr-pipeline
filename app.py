#!/usr/bin/env python3
"""
Web UI for Bangla & English ASR powered by Whisper (large-v3-turbo / tiny).
Features:
- Live microphone recording & audio upload
- Smart Bilingual Language Routing (Indic -> Bangla, English -> English)
- Timestamped segments breakdown
- Batch folder transcription
- Dataset WER/CER evaluation benchmark
"""

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR / "scripts"))

from transcribe import get_transcriber, transcribe_file
from evaluate import compute_metrics

# Global model cache to prevent reloading large weights on every inference
MODEL_CACHE = {}

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

def transcribe_audio_ui(audio_path, model_name, language_choice, beam_size):
    """Processes uploaded or recorded audio and returns structured results."""
    if not audio_path:
        return "Please upload an audio file or record speech via microphone.", "", None, None

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    try:
        model = load_cached_model(model_name)
        res = transcribe_file(model, audio_path, language=lang_code, beam_size=beam_size)

        info_md = f"""
### 📊 Transcription Details
- **Detected Language:** `{res['language']}` ({res['language_probability']:.1%})
- **Audio Duration:** `{res['duration_sec']}s`
- **Processing Time:** `{res['transcription_time_sec']}s` (Speed: **{res['speed_factor']}x** real-time)
"""

        segments_data = [
            {"Start (s)": s["start"], "End (s)": s["end"], "Transcription": s["text"]}
            for s in res["segments"]
        ]
        df_segments = pd.DataFrame(segments_data)

        return res["text"], info_md, df_segments, json.dumps(res, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Error during transcription: {str(e)}", "", None, None

def batch_transcribe_ui(directory_path, model_name, language_choice):
    """Transcribes an entire directory of audio files."""
    dir_p = Path(directory_path)
    if not dir_p.exists() or not dir_p.is_dir():
        return f"Error: Directory '{directory_path}' does not exist.", None, None

    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [p for p in dir_p.rglob("*") if p.suffix.lower() in valid_exts]

    if not audio_files:
        return f"No audio files found in '{directory_path}'.", None, None

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    model = load_cached_model(model_name)
    results = []

    for f in audio_files:
        try:
            res = transcribe_file(model, f, language=lang_code)
            results.append({
                "File": f.name,
                "Language": res["language"],
                "Duration (s)": res["duration_sec"],
                "Transcription": res["text"]
            })
        except Exception as err:
            results.append({
                "File": f.name,
                "Language": "Error",
                "Duration (s)": 0,
                "Transcription": str(err)
            })

    df = pd.DataFrame(results)
    out_json = str(SCRIPT_DIR / "batch_transcriptions.json")
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)

    summary_text = f"Successfully transcribed {len(results)} files from `{directory_path}`."
    return summary_text, df, out_json

def evaluate_dataset_ui(metadata_csv, audio_dir, model_name, language_choice):
    """Calculates WER & CER benchmark on test set."""
    csv_p = Path(metadata_csv)
    if not csv_p.exists():
        return f"Error: CSV file '{metadata_csv}' not found.", "", None

    df = pd.read_csv(csv_p)
    audio_col = next((c for c in ["audio_path", "audio", "file_name", "path"] if c in df.columns), None)
    text_col = next((c for c in ["sentence", "transcription", "ground_truth", "text"] if c in df.columns), None)

    if not audio_col or not text_col:
        return f"Error: CSV must have audio path and text columns. Detected: {list(df.columns)}", "", None

    lang_code = {
        "Auto (Smart Bilingual: Bangla / English)": "auto",
        "Bangla (বাংলা)": "bn",
        "English (en)": "en"
    }.get(language_choice, "auto")

    try:
        import jiwer
    except ImportError:
        return "Error: jiwer is required for evaluation.", "", None

    model = load_cached_model(model_name)
    audio_base = Path(audio_dir) if audio_dir else csv_p.parent

    eval_rows = []
    for _, row in df.iterrows():
        rel_audio = str(row[audio_col]).strip()
        ref_text = str(row[text_col]).strip()

        audio_full = audio_base / rel_audio
        if not audio_full.is_file() and Path(rel_audio).is_file():
            audio_full = Path(rel_audio)

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
        return "No audio files were found or evaluated.", "", None

    results_df = pd.DataFrame(eval_rows)
    all_refs = [r["Ground Truth"] for r in eval_rows]
    all_hyps = [r["Prediction"] for r in eval_rows]
    overall_wer = jiwer.wer(all_refs, all_hyps)
    overall_cer = jiwer.cer(all_refs, all_hyps)

    metric_md = f"""
## 🏆 Benchmark Evaluation Results
- **Total Samples Evaluated:** `{len(results_df)}`
- **Overall Word Error Rate (WER):** `{overall_wer:.2%}`
- **Overall Character Error Rate (CER):** `{overall_cer:.2%}`
"""
    return "Evaluation completed successfully!", metric_md, results_df

# ==============================================================================
# Gradio UI Construction
# ==============================================================================
custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate"
)

with gr.Blocks(title="Bangla & English ASR - Whisper") as demo:
    gr.Markdown("""
# 🎙️ Bangla & English Automatic Speech Recognition (ASR)
### Powered by Whisper `large-v3-turbo` with Smart Bilingual Routing (বাংলা & English)
""")

    with gr.Tabs():
        # TAB 1: Single Audio & Microphone
        with gr.TabItem("🎯 Live Mic & Audio File Transcription"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record Speech or Upload Audio File (.wav, .mp3, .flac)"
                    )

                    with gr.Accordion("⚙️ Model & Language Settings", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=["large-v3-turbo", "tiny"],
                            value="large-v3-turbo",
                            label="Whisper Model Checkpoint"
                        )
                        lang_dropdown = gr.Dropdown(
                            choices=[
                                "Auto (Smart Bilingual: Bangla / English)",
                                "Bangla (বাংলা)",
                                "English (en)"
                            ],
                            value="Auto (Smart Bilingual: Bangla / English)",
                            label="Language Mode"
                        )
                        beam_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Beam Size (Decoding Accuracy)"
                        )

                    transcribe_btn = gr.Button("🚀 Transcribe Audio", variant="primary", size="lg")

                    # Example Audio Files
                    examples = [
                        ["data/test/audio/sample_test1.mp3", "large-v3-turbo", "Auto (Smart Bilingual: Bangla / English)", 5],
                        ["data/test/audio/complete_Actor - 46_03-01-06-02-04-03-46.wav", "large-v3-turbo", "Auto (Smart Bilingual: Bangla / English)", 5],
                        ["data/test/audio/sample_test2.mp3", "large-v3-turbo", "Auto (Smart Bilingual: Bangla / English)", 5]
                    ]
                    available_examples = [ex for ex in examples if Path(ex[0]).exists()]
                    if available_examples:
                        gr.Examples(
                            examples=available_examples,
                            inputs=[audio_input, model_dropdown, lang_dropdown, beam_slider],
                            label="Quick Test Samples"
                        )

                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="📝 Transcribed Text",
                        lines=6,
                        placeholder="Transcription will appear here..."
                    )
                    metrics_output = gr.Markdown()

                    with gr.Accordion("⏱️ Timestamped Segments Breakdown", open=False):
                        segments_table = gr.DataFrame(
                            headers=["Start (s)", "End (s)", "Transcription"],
                            label="Segment Details",
                            wrap=True
                        )

                    with gr.Accordion("🔍 Raw JSON Metadata", open=False):
                        json_output = gr.Code(language="json", label="JSON Output")

            transcribe_btn.click(
                fn=transcribe_audio_ui,
                inputs=[audio_input, model_dropdown, lang_dropdown, beam_slider],
                outputs=[output_text, metrics_output, segments_table, json_output]
            )

        # TAB 2: Batch Directory Transcription
        with gr.TabItem("📂 Batch Directory Transcription"):
            gr.Markdown("### Transcribe all audio files inside a specified directory")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_dir_input = gr.Textbox(
                        value="data/test/audio",
                        label="Directory Path containing Audio Files"
                    )
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
                    batch_btn = gr.Button("⚡ Start Batch Transcription", variant="primary")

                with gr.Column(scale=1):
                    batch_status = gr.Markdown()
                    batch_download = gr.File(label="Download Full Results (JSON)")

            batch_table = gr.DataFrame(
                headers=["File", "Language", "Duration (s)", "Transcription"],
                label="Batch Transcription Results",
                wrap=True
            )

            batch_btn.click(
                fn=batch_transcribe_ui,
                inputs=[batch_dir_input, batch_model, batch_lang],
                outputs=[batch_status, batch_table, batch_download]
            )

        # TAB 3: Dataset Benchmark (WER & CER)
        with gr.TabItem("📊 Benchmark & Evaluate (WER / CER)"):
            gr.Markdown("### Benchmark Whisper accuracy on test datasets with Ground Truth transcriptions")
            with gr.Row():
                with gr.Column(scale=1):
                    eval_csv_input = gr.Textbox(
                        value="data/test/metadata.csv",
                        label="Metadata CSV File Path"
                    )
                    eval_audio_dir = gr.Textbox(
                        value="data/test/audio",
                        label="Audio Directory Path"
                    )
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
                    eval_btn = gr.Button("📈 Compute WER & CER", variant="primary")

                with gr.Column(scale=1):
                    eval_status = gr.Markdown()
                    eval_metrics_display = gr.Markdown()

            eval_results_table = gr.DataFrame(
                headers=["File", "Ground Truth", "Prediction", "WER", "CER"],
                label="Detailed Sample-Level Predictions & Errors",
                wrap=True
            )

            eval_btn.click(
                fn=evaluate_dataset_ui,
                inputs=[eval_csv_input, eval_audio_dir, eval_model, eval_lang],
                outputs=[eval_status, eval_metrics_display, eval_results_table]
            )

if __name__ == "__main__":
    # Launch locally on port 7860
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
