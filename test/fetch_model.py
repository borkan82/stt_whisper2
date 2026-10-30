#!/usr/bin/env python3
"""Download a faster-whisper model into /app/models/<dir-name>.

  fetch_model.py large-v3-turbo      -> /app/models/faster-whisper-large-v3-turbo
  fetch_model.py large-v3            -> /app/models/faster-whisper-large-v3
"""
import sys
from huggingface_hub import snapshot_download
from faster_whisper.utils import _MODELS

name = sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo"
repo = _MODELS.get(name, name)
target = f"/app/models/faster-whisper-{name}" if name in _MODELS else f"/app/models/{repo.split('/')[-1]}"
print(f"downloading {repo} -> {target}", flush=True)
snapshot_download(repo, local_dir=target,
                  allow_patterns=["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"])
print("done")
