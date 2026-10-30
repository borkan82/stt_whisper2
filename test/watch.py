#!/usr/bin/env python3
"""
Watch-folder service for `docker compose up -d`.

Drop an audio file into ./audio -> transcripts appear in ./out:
  <name>.baseline.<fmt>   stock single-language transcribe()
  <name>.perturn.<fmt>    per-turn language detection (issue #73)

Env:
  MODEL            faster-whisper alias or local dir (default large-v3-turbo)
  MODES            baseline,perturn | perturn | baseline (default both)
  FORMAT           srt | vtt | text (default srt)
  DEVICE           cpu | cuda (default cpu)
  COMPUTE_TYPE     int8 on CPU, float16 on CUDA (default int8)
  CPU_THREADS      default: all cores
  POLL_SECONDS     default 5
  STT_WHISPER2_VAD_* / STT_WHISPER2_PER_TURN_*   same meaning as in the ExApp

A file is processed once; delete its outputs in ./out to process it again.
A file is only picked up after its size stops changing (safe while copying).
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from faster_whisper import WhisperModel  # noqa: E402
from compare import per_turn, run_one, vad_settings  # noqa: E402  (compare puts lib/ on sys.path)

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm", ".mp4", ".mkv", ".aac", ".wma"}

AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "/audio"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/out"))
MODEL = os.environ.get("MODEL", "large-v3-turbo")
MODES = [m.strip() for m in os.environ.get("MODES", "baseline,perturn").split(",") if m.strip()]
FORMAT = os.environ.get("FORMAT", "srt")
DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.environ.get("CPU_THREADS", "0") or 0) or (os.cpu_count() or 4)
POLL = float(os.environ.get("POLL_SECONDS", "5"))


def pending(sizes: dict) -> list[Path]:
    """Files whose outputs are missing and whose size was stable since the last poll."""
    ready = []
    for f in sorted(AUDIO_DIR.iterdir()) if AUDIO_DIR.exists() else []:
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXT:
            continue
        if all((OUT_DIR / f"{f.stem}.{m}.{FORMAT}").exists() for m in MODES):
            continue
        size = f.stat().st_size
        if sizes.get(f) == size and size > 0:
            ready.append(f)
        sizes[f] = size
    return ready


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger(per_turn.LOGGER.name).setLevel(logging.DEBUG)
    bad = [m for m in MODES if m not in ("baseline", "perturn")]
    if bad or FORMAT not in ("srt", "vtt", "text"):
        sys.exit(f"invalid MODES={MODES} or FORMAT={FORMAT}")

    vad_filter, vad_parameters = vad_settings(os.environ.get("STT_WHISPER2_VAD_FILTER", "1") == "1")
    print(f"loading model {MODEL} on {DEVICE} ({COMPUTE_TYPE}, {CPU_THREADS} threads) "
          f"- first start downloads it", flush=True)
    model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    print(f"ready: watching {AUDIO_DIR} -> {OUT_DIR}, modes={MODES}, format={FORMAT}", flush=True)

    sizes: dict = {}
    while True:
        for f in pending(sizes):
            for mode in MODES:
                if (OUT_DIR / f"{f.stem}.{mode}.{FORMAT}").exists():
                    continue
                try:
                    run_one(model, str(f), mode, FORMAT, OUT_DIR, vad_filter, vad_parameters)
                except Exception as e:  # keep the service alive; write the error next to the outputs
                    msg = "".join(traceback.format_exception(e))
                    print(f"FAILED {f.name} ({mode}):\n{msg}", flush=True)
                    (OUT_DIR / f"{f.stem}.{mode}.{FORMAT}").write_text(f"ERROR\n{msg}", encoding="utf-8")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
