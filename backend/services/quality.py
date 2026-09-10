from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


TranscriptionProfile = Literal["auto", "balanced", "bangla_high_accuracy", "english_fast", "fast"]
VadAggressiveness = Literal["off", "low", "medium", "high"]


BANGLA_DOMAIN_PROMPT = (
    "বাংলা ভাষার স্পষ্ট প্রতিলিপি লিখুন। বক্তৃতা, সংবাদ, সংসদ, কৃষি, রাজনীতি, "
    "বাংলাদেশ, সার, কৃষক, মন্ত্রী, স্পিকার, সরকার, বিরোধী দল, জেলা, উপজেলা।"
)


@dataclass(frozen=True)
class ResolvedQuality:
    profile: str
    transcribe_kwargs: dict[str, Any]


def resolve_profile(profile: str | None, language: str | None) -> str:
    if profile and profile != "auto":
        return "english_fast" if profile == "fast" else profile
    if language == "bn":
        return "bangla_high_accuracy"
    if language == "en":
        return "english_fast"
    return "balanced"


def vad_parameters(vad_filter: bool, aggressiveness: str = "medium") -> dict[str, Any] | None:
    if not vad_filter or aggressiveness == "off":
        return None
    presets: dict[str, dict[str, Any]] = {
        "low": {"min_silence_duration_ms": 800, "speech_pad_ms": 300},
        "medium": {"min_silence_duration_ms": 500, "speech_pad_ms": 250},
        "high": {"min_silence_duration_ms": 300, "speech_pad_ms": 200},
    }
    return presets.get(aggressiveness, presets["medium"])


def resolve_transcription_quality(
    *,
    profile: str | None,
    language: str | None,
    beam_size: int,
    temperature: float,
    vad_filter: bool,
    vad_aggressiveness: str,
    condition_on_previous_text: bool,
    chunk_length: int | None,
    initial_prompt: str | None,
    hotwords: str | None,
) -> ResolvedQuality:
    resolved = resolve_profile(profile, language)
    kwargs: dict[str, Any] = {
        "beam_size": beam_size,
        "vad_filter": vad_filter and vad_aggressiveness != "off",
        "temperature": temperature,
        "condition_on_previous_text": condition_on_previous_text,
    }
    params = vad_parameters(vad_filter, vad_aggressiveness)
    if params:
        kwargs["vad_parameters"] = params
    if chunk_length and chunk_length > 0:
        kwargs["chunk_length"] = chunk_length
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt.strip()
    if hotwords:
        kwargs["hotwords"] = hotwords.strip()

    if resolved == "bangla_high_accuracy":
        kwargs.update(
            {
                "beam_size": max(beam_size, 8),
                "best_of": max(beam_size, 8),
                "patience": 1.2,
                "temperature": [0.0, 0.2, 0.4],
                "compression_ratio_threshold": 2.0,
                "log_prob_threshold": -0.8,
                "no_speech_threshold": 0.45,
                "condition_on_previous_text": False,
                "prompt_reset_on_temperature": 0.2,
                "repetition_penalty": 1.08,
                "no_repeat_ngram_size": 4,
                "hallucination_silence_threshold": 2.0,
                "chunk_length": chunk_length or 30,
            }
        )
        prompt = " ".join(part for part in [BANGLA_DOMAIN_PROMPT, initial_prompt or ""] if part).strip()
        kwargs["initial_prompt"] = prompt
        kwargs["hotwords"] = " ".join(part for part in [hotwords or "", "বাংলাদেশ সংসদ কৃষক সার সরকার স্পিকার"] if part).strip()
    elif resolved == "english_fast":
        kwargs.update(
            {
                "beam_size": min(beam_size, 5),
                "best_of": min(max(beam_size, 1), 5),
                "patience": 1,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "chunk_length": chunk_length or 30,
            }
        )
    else:
        kwargs.update(
            {
                "best_of": max(beam_size, 5),
                "patience": 1,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "chunk_length": chunk_length or 30,
            }
        )

    return ResolvedQuality(profile=resolved, transcribe_kwargs=kwargs)


def cleanup_text(text: str) -> str:
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([।!?])\s*\1+", r"\1", text)
    return text.strip()


def repeated_token_ratio(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 3:
        return 0.0
    repeated = sum(1 for previous, current in zip(tokens, tokens[1:]) if previous == current)
    return repeated / max(1, len(tokens) - 1)


def collapse_repeated_runs(text: str, max_run: int = 3) -> tuple[str, bool]:
    tokens = text.split()
    if not tokens:
        return text, False
    changed = False
    collapsed: list[str] = []
    last = None
    run = 0
    for token in tokens:
        if token == last:
            run += 1
        else:
            last = token
            run = 1
        if run <= max_run:
            collapsed.append(token)
        else:
            changed = True
    return " ".join(collapsed), changed


def postprocess_result(result: dict[str, Any], profile: str, repetition_guard: bool = True) -> dict[str, Any]:
    warnings: list[str] = []
    segments = []
    suspicious_segments = 0
    for segment in result.get("segments", []):
        text = cleanup_text(str(segment.get("text", "")))
        ratio = repeated_token_ratio(text)
        repeated = ratio >= 0.18
        if repeated:
            suspicious_segments += 1
            warnings.append(f"Repeated phrase pattern around {segment.get('start')}s-{segment.get('end')}s.")
        if repetition_guard:
            text, changed = collapse_repeated_runs(text)
            repeated = repeated or changed
        segments.append({**segment, "text": text, "repetition_score": round(ratio, 4), "suspicious": repeated})

    full_text = cleanup_text(" ".join(str(segment.get("text", "")).strip() for segment in segments))
    full_ratio = repeated_token_ratio(full_text)
    if full_ratio >= 0.12:
        warnings.append("Transcript contains repeated phrase patterns; try Bangla High Accuracy with shorter chunks.")
    if repetition_guard:
        full_text, changed = collapse_repeated_runs(full_text)
        if changed:
            warnings.append("Repeated token runs were trimmed from the transcript.")

    lang_prob = float(result.get("language_probability") or 0)
    language = str(result.get("language") or "")
    low_confidence = lang_prob < 0.6
    if low_confidence:
        warnings.append("Low language confidence; force Bangla and use Bangla High Accuracy for this audio.")

    result["segments"] = segments
    result["text"] = full_text
    result["quality"] = {
        "profile": profile,
        "language_probability": round(lang_prob, 4),
        "low_confidence": low_confidence,
        "repetition_score": round(full_ratio, 4),
        "suspicious_segment_count": suspicious_segments,
        "warnings": list(dict.fromkeys(warnings)),
        "suggested_retry": "Use language=bn, profile=bangla_high_accuracy, chunk_length=20-30, and condition_on_previous_text=false."
        if low_confidence or full_ratio >= 0.12 or suspicious_segments
        else "",
        "applies_bangla_cleanup": "bn" in language.lower() or profile == "bangla_high_accuracy",
    }
    return result
