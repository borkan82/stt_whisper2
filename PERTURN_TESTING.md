# Testing per-turn language detection (issue #73)

Adds `STT_WHISPER2_PER_TURN_LANGUAGE` (default `0`). When `1`, the file is split at
pauses with Silero VAD, the language is detected per turn, same-language turns are
merged into spans, and each span is transcribed with `language=` fixed. Timestamps
are shifted back to recording time, so text/SRT/VTT output keeps its shape.

Changed files: `lib/per_turn.py` (new), `lib/main.py` (one call site), `appinfo/info.xml`
(4 env vars). Test helpers: `Dockerfile.cpu`, `docker-compose.yml`, `.env.example`, `test/watch.py`,
`test/compare.py`, `test/fetch_model.py`, `test/make_mixed.sh`.

| Env var | Default | Meaning |
|---|---|---|
| `STT_WHISPER2_PER_TURN_LANGUAGE` | `0` | `1` enables the per-turn path |
| `STT_WHISPER2_PER_TURN_MIN_SILENCE_MS` | `1000` | pause that separates turns |
| `STT_WHISPER2_PER_TURN_MIN_DETECT_S` | `3` | shorter turns inherit the previous language |
| `STT_WHISPER2_PER_TURN_MIN_PROB` | `0.6` | less certain turns inherit the previous language |

## 1. Quick start: `docker compose up -d`

```bash
cp .env.example .env          # optional: model, output format, tuning
docker compose up -d --build  # starts the watcher (first start downloads the model)
docker compose logs -f watcher

cp ~/recordings/meeting.mp3 audio/
# -> out/meeting.baseline.srt   stock transcribe()
# -> out/meeting.perturn.srt    per-turn language detection
```

Each file is processed once. Delete its outputs in `out/` to run it again. After changing `.env`,
run `docker compose up -d` again. The logs show the detected language per chunk, the resulting
spans, and the speed (RTF) for both modes.

### One-shot A/B on a single file

```bash
DC="docker compose --profile tools"
$DC run --rm compare /audio/meeting.mp3 --model large-v3 --format text
$DC run --rm compare /audio/meeting.mp3 --mode perturn --format vtt
```

GPU: set `DEVICE=cuda` and `COMPUTE_TYPE=float16`. This needs the upstream CUDA `Dockerfile` instead of `Dockerfile.cpu`, plus `gpus: all`.

### Make a mixed-language clip

```bash
# ./audio/es1.mp3 ./audio/en.mp3 ./audio/es2.mp3 -> ./audio/mixed.wav (1.5 s pauses)
docker compose --profile tools run --rm make-mixed /audio/es1.mp3 /audio/en.mp3 /audio/es2.mp3
docker compose --profile tools run --rm -e PAUSE=0.8 -e OUT=/audio/short-pauses.wav make-mixed /audio/a.mp3 /audio/b.mp3
```

Worth testing beyond the issue's case: pauses shorter than 1 s between languages
(expect them to merge — tune `MIN_SILENCE_MS`), short interjections ("ok", "ja"),
three languages, and a single-language file (output should match baseline).

## 2. Inside Nextcloud (manual_install ExApp)

```bash
docker compose --profile tools run --rm fetch-model large-v3-turbo     # -> ./models/faster-whisper-large-v3-turbo
# set NEXTCLOUD_URL and APP_SECRET in .env, then
docker compose --profile exapp up -d exapp
```

Register it in Nextcloud (adjust the container name; `manual_install` daemon must exist —
admin settings → AppAPI → register a "Manual install" daemon pointing at the Docker host):

```bash
docker exec -u www-data <nextcloud-container> php occ app_api:app:unregister stt_whisper2 --silent --force
docker exec -u www-data <nextcloud-container> php occ app_api:app:register stt_whisper2 manual_install \
  --json-info '{"id":"stt_whisper2","name":"Local Whisper Speech To Text","daemon_config_name":"manual_install","version":"2.5.0","secret":"12345","port":9030}' \
  --wait-finish
```

Nextcloud must reach the container on port 9030 and the container must reach `NEXTCLOUD_URL`.
Then transcribe a file via Assistant / a Talk recording and check `docker logs -f` for the
`per-turn spans:` line. Flip `STT_WHISPER2_PER_TURN_LANGUAGE=0` and `up -d` again for the baseline.
