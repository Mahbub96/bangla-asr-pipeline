#!/usr/bin/env python3
"""Segmented curl downloader for SUBAK.KO Hugging Face dataset files.

Downloads TSV metadata and parquet shards from:
https://huggingface.co/datasets/SUST-CSE-Speech/SUBAK.KO

Each large file is split into byte ranges and fetched by multiple curl
processes, similar to IDM-style segmented downloading. Completed parts are
combined into the final file after size verification.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import urlopen

API_URL = "https://huggingface.co/api/datasets/SUST-CSE-Speech/SUBAK.KO"
BASE_RESOLVE = "https://huggingface.co/datasets/SUST-CSE-Speech/SUBAK.KO/resolve/main"
DEFAULT_OUT = Path("data/sources/subakko/hf")


def run(cmd: list[str], *, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def fetch_file_list() -> list[str]:
    with urlopen(API_URL, timeout=60) as response:
        payload = json.load(response)
    files = [item["rfilename"] for item in payload["siblings"]]
    wanted = [
        name
        for name in files
        if (name.startswith("data/") and name.endswith(".parquet"))
        or (name.startswith("Data/") and name.endswith(".tsv"))
        or name == "README.md"
    ]
    return sorted(wanted)


def hf_url(path: str) -> str:
    return f"{BASE_RESOLVE}/{quote(path)}?download=true"


def remote_size(url: str) -> int:
    # -L follows the Hugging Face signed redirect; -w prints final size.
    result = run(["curl", "-L", "-sS", "-I", "-o", "/dev/null", "-w", "%{size_download} %{http_code} %{url_effective}", url])
    # Some servers do not expose size via HEAD size_download. Fall back to headers.
    head = run(["curl", "-L", "-sS", "-I", url])
    if head.returncode != 0:
        raise RuntimeError(head.stderr.strip() or f"HEAD failed: {url}")
    content_lengths = []
    for line in head.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                content_lengths.append(int(line.split(":", 1)[1].strip()))
            except ValueError:
                pass
    if not content_lengths:
        raise RuntimeError(f"Could not determine remote size for {url}")
    return content_lengths[-1]


def ranges_for(size: int, segments: int) -> list[tuple[int, int]]:
    if size <= 0:
        return []
    segments = max(1, min(segments, size))
    chunk = (size + segments - 1) // segments
    ranges = []
    start = 0
    while start < size:
        end = min(size - 1, start + chunk - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def download_part(url: str, part_path: Path, start: int, end: int, retries: int) -> None:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(part_path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        if tmp.exists():
            tmp.unlink()
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--range",
            f"{start}-{end}",
            "-o",
            str(tmp),
            url,
        ]
        result = run(cmd, quiet=True)
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size == expected:
            tmp.replace(part_path)
            return
        got = tmp.stat().st_size if tmp.exists() else 0
        print(
            f"part retry {attempt}/{retries}: {part_path.name} expected={expected} got={got} err={result.stderr.strip()[:200]}",
            flush=True,
        )
        time.sleep(min(10, attempt * 2))
    raise RuntimeError(f"failed part {part_path} bytes={start}-{end}")


def combine_parts(parts: Iterable[Path], final_path: Path, expected_size: int) -> None:
    tmp = final_path.with_suffix(final_path.suffix + ".assembling")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
    if tmp.stat().st_size != expected_size:
        raise RuntimeError(f"assembled size mismatch for {final_path}: {tmp.stat().st_size} != {expected_size}")
    tmp.replace(final_path)


def download_file(path: str, out_root: Path, segments: int, retries: int) -> None:
    url = hf_url(path)
    final_path = out_root / path
    final_path.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(url)
    if final_path.exists() and final_path.stat().st_size == size:
        print(f"SKIP complete {path} ({size} bytes)", flush=True)
        return
    if final_path.exists():
        print(f"REPAIR {path}: local={final_path.stat().st_size} remote={size}", flush=True)
        final_path.unlink()

    use_segments = 1 if size < 8 * 1024 * 1024 else segments
    byte_ranges = ranges_for(size, use_segments)
    parts_dir = final_path.with_name(final_path.name + ".parts")
    parts = [parts_dir / f"part-{idx:03d}" for idx in range(len(byte_ranges))]

    print(f"START {path} size={size} segments={len(byte_ranges)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(byte_ranges)) as pool:
        futures = [
            pool.submit(download_part, url, part, start, end, retries)
            for part, (start, end) in zip(parts, byte_ranges)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    combine_parts(parts, final_path, size)
    for part in parts:
        part.unlink(missing_ok=True)
    try:
        parts_dir.rmdir()
    except OSError:
        pass
    print(f"DONE {path} ({size} bytes)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Segmented curl downloader for SUBAK.KO")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--segments", type=int, default=8, help="Parallel byte ranges per large file")
    parser.add_argument("--files", type=int, default=2, help="Parallel files")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--split", choices=["all", "train", "validation", "test", "meta"], default="all")
    args = parser.parse_args()

    selected = fetch_file_list()
    if args.split == "meta":
        selected = [p for p in selected if p.startswith("Data/") or p == "README.md"]
    elif args.split != "all":
        selected = [p for p in selected if p.startswith(f"data/{args.split}-") or p in {f"Data/{args.split}.tsv", "README.md"}]

    print(f"Downloading {len(selected)} SUBAK.KO files to {args.out}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.files) as pool:
        futures = {
            pool.submit(download_file, path, args.out, args.segments, args.retries): path
            for path in selected
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:  # keep other files going
                failures.append((path, str(exc)))
                print(f"FAIL {path}: {exc}", flush=True)

    if failures:
        print("FAILED FILES:", flush=True)
        for path, error in failures:
            print(f"- {path}: {error}", flush=True)
        return 1
    print("SUBAK.KO download complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
