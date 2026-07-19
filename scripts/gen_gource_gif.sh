#!/usr/bin/env bash
# Render the repository history to an animated GIF with gource, for the README.
#
#   scripts/gen_gource_gif.sh
#
# Needs: gource, ffmpeg, gifsicle, and a display. On a headless machine wrap the
# whole call in xvfb-run (the script does this automatically when $DISPLAY is
# unset). Writes docs/media/history.gif. The GIF is committed so the README
# renders on GitHub; regenerate it on demand (see .github/workflows or `make media`).
set -euo pipefail

cd "$(dirname "$0")/.."
export OUT=docs/media/history.gif
export TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

render() {
    gource \
        --seconds-per-day 0.15 --auto-skip-seconds 0.1 --max-file-lag 0.1 \
        --hide filenames,mouse,progress --key --title "AutStr" \
        -1280x720 --stop-at-end --output-framerate 30 \
        --background-colour 0d1117 --bloom-multiplier 0.35 \
        --output-ppm-stream - \
    | ffmpeg -y -loglevel error -r 30 -f image2pipe -vcodec ppm -i - \
        -vcodec ffv1 "$TMP/history.mkv"      # lossless: ffmpeg-free has no libx264
}

# gource needs an OpenGL context; supply a virtual display if there is none.
if [ -z "${DISPLAY:-}" ]; then
    xvfb-run -a -s "-screen 0 1280x720x24" bash -c "$(declare -f render); render"
else
    render
fi

# lossless intermediate -> palette-optimised GIF (small, GitHub-friendly)
FILT="fps=10,scale=640:-1:flags=lanczos"
ffmpeg -y -loglevel error -i "$TMP/history.mkv" \
    -vf "$FILT,palettegen=max_colors=128:stats_mode=diff" "$TMP/pal.png"
ffmpeg -y -loglevel error -i "$TMP/history.mkv" -i "$TMP/pal.png" \
    -lavfi "$FILT[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" "$OUT"
gifsicle -O3 --lossy=80 --colors 128 "$OUT" -o "$OUT"

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
