#!/bin/bash
# Concatenate clips with a pause between them into one 16 kHz mono test file.
#   make_mixed.sh /audio/es1.mp3 /audio/en.mp3 /audio/es2.mp3
# Env: PAUSE (seconds, default 1.5), OUT (default /audio/mixed.wav)
set -e
PAUSE="${PAUSE:-1.5}"; OUT="${OUT:-/audio/mixed.wav}"
[ $# -ge 2 ] || { echo "usage: make_mixed.sh clip1 clip2 [clip3 ...]"; exit 1; }
args=(); filter=""; seq=""; idx=0; n=$#; k=0
for f in "$@"; do
  args+=(-i "$f")
  filter+="[$idx:a]aresample=16000,aformat=channel_layouts=mono[c$idx];"
  seq+="[c$idx]"; idx=$((idx+1)); k=$((k+1))
  if [ $k -lt $n ]; then
    args+=(-f lavfi -t "$PAUSE" -i "anullsrc=r=16000:cl=mono")
    seq+="[$idx:a]"; idx=$((idx+1))
  fi
done
ffmpeg -hide_banner -loglevel error -y "${args[@]}" \
  -filter_complex "${filter}${seq}concat=n=$idx:v=0:a=1[out]" -map "[out]" -ar 16000 -ac 1 "$OUT"
echo "written $OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")s)"
