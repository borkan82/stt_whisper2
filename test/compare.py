#!/usr/bin/env python3
"""
Compare the stock single-language transcription with the per-turn language path
(nextcloud/stt_whisper2#73) on one audio file — no Nextcloud needed.

  python3 compare.py /audio/meeting.mp3 --model large-v3-turbo
  python3 compare.py /audio/meeting.mp3 --model /app/models/faster-whisper-large-v3 --mode perturn --format srt

--model accepts a faster-whisper alias (tiny, base, small, medium, large-v3,
large-v3-turbo, ...) which is downloaded to the HF cache, or a local model dir.
Results are printed and written to --out as <file>.<mode>.<format>.
"""

import argparse
import logging
import os
import sys
from math import floor, modf
from pathlib import Path
from time import gmtime, perf_counter, strftime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from faster_whisper import WhisperModel  # noqa: E402
import per_turn  # noqa: E402


def ts(t: float, sep: str) -> str:
    frac, whole = modf(t)
    return f"{strftime('%H:%M:%S', gmtime(whole))}{sep}{floor(frac * 1000):03d}"


def render(segments, fmt: str) -> str:
    if fmt == "text":
        return "".join(s.text for s in segments).strip()
    out = ["WEBVTT\n"] if fmt == "vtt" else []
    sep = "." if fmt == "vtt" else ","
    for i, s in enumerate(segments, 1):
        head = "" if fmt == "vtt" else f"{i}\n"
        out.append(f"{head}{ts(s.start, sep)} --> {ts(s.end, sep)}\n{s.text.strip()}\n")
    return "\n".join(out)


def vad_settings(enabled: bool = True):
    """Same VAD defaults as lib/main.py."""
    if not enabled:
        return False, None
    return True, {
        "threshold": float(os.environ.get("STT_WHISPER2_VAD_THRESHOLD", "0.5")),
        "min_speech_duration_ms": int(os.environ.get("STT_WHISPER2_VAD_MIN_SPEECH_MS", "0")),
        "min_silence_duration_ms": int(os.environ.get("STT_WHISPER2_VAD_MIN_SILENCE_MS", "2000")),
        "speech_pad_ms": int(os.environ.get("STT_WHISPER2_VAD_SPEECH_PAD_MS", "400")),
    }


def run_one(model, audio: str, mode: str, fmt: str, out_dir: Path, vad_filter, vad_parameters) -> Path:
    """Transcribe one file in one mode, print the result and write <out>/<stem>.<mode>.<fmt>."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    if mode == "baseline":
        segments, info = model.transcribe(audio, vad_filter=vad_filter, vad_parameters=vad_parameters)
    else:
        segments, info = per_turn.transcribe_per_turn(
            model, audio, vad_filter=vad_filter, vad_parameters=vad_parameters
        )
    segments = list(segments)  # transcription actually runs here
    elapsed = perf_counter() - t0
    text = render(segments, fmt)
    print(f"\n===== {Path(audio).name} | {mode.upper()} | detected/first language: {info.language} "
          f"| {elapsed:.1f}s for {info.duration:.1f}s audio "
          f"(RTF {elapsed / max(info.duration, 0.001):.2f}) =====")
    print(text, flush=True)
    path = out_dir / f"{Path(audio).stem}.{mode}.{fmt}"
    path.write_text(text + "\n", encoding="utf-8")
    print(f"----- written to {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--mode", choices=["baseline", "perturn", "both"], default="both")
    ap.add_argument("--format", choices=["text", "srt", "vtt"], default="srt")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--compute-type", default="int8", help="int8 for CPU, float16 for CUDA")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--no-vad", action="store_true", help="disable the VAD filter inside transcribe()")
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/out"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger(per_turn.LOGGER.name).setLevel(logging.DEBUG)

    vad_filter, vad_parameters = vad_settings(not args.no_vad)

    print(f"loading model {args.model} on {args.device} ({args.compute_type}, {args.threads} threads)", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=args.threads)

    modes = ["baseline", "perturn"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_one(model, args.audio, mode, args.format, Path(args.out), vad_filter, vad_parameters)


if __name__ == "__main__":
    main()
