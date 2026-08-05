#!/usr/bin/env bash
# Command sequence for the iqforge demo recording.
#
# Written to be captured with asciinema; the recording instructions live in
# docs/demo.md. The script needs no interaction, it runs start to finish.
#
# Idempotent: output directories are removed at the start of every run, so a
# re-recording never shows "directory already exists" noise.

set -uo pipefail

# Fixed terminal width: rich reads COLUMNS, so the tables and the spectrogram
# always look the same regardless of the recording width.
export COLUMNS=100
export LINES=34

# The spectrogram uses the upper half block (U+2580) and the tables use box
# drawing characters. When output is redirected to a file Python falls back to
# the local code page and those characters are lost; force UTF-8.
export PYTHONIOENCODING=utf-8

DATASET_DIR="${DATASET_DIR:-/tmp/ds}"
SINGLE_DIR="${SINGLE_DIR:-/tmp/ds-single}"

# Pick the runner fastest first. `uv run` re-resolves the environment on every
# call, adding about 2 seconds per command; across six commands that is 12
# seconds of needless recording. Use the binary directly when there is one.
if command -v iqforge >/dev/null 2>&1; then
    IQ=(iqforge)
elif [ -x .venv/bin/iqforge ]; then
    IQ=(.venv/bin/iqforge)
elif [ -x .venv/Scripts/iqforge.exe ]; then
    IQ=(.venv/Scripts/iqforge.exe)
else
    IQ=(uv run iqforge)
fi

rm -rf "$DATASET_DIR" "$SINGLE_DIR"

# Echo the command, run it, then pause for the given number of seconds.
run() {
    local pause="$1"
    shift
    printf '\n\033[1;32m$\033[0m \033[1miqforge %s\033[0m\n\n' "$*"
    "${IQ[@]}" "$@"
    sleep "$pause"
}

# 1. What is in the recording?
run 1.5 info examples/bpsk_01.sigmf-meta

# 2. Look at the signal. Longer pause so the spectrogram can be read.
run 4 inspect examples/bpsk_01.sigmf-meta

# 3. Window, label, split, write.
run 1.5 build examples/ -o "$DATASET_DIR" --balance-by core:freq_lower_edge

# 4. What did we just build?
run 2.5 stats "$DATASET_DIR"

# 5. Is the dataset actually trainable?
run 2 train "$DATASET_DIR" --epochs 20

# 6. The point: it does not quietly do what it cannot do.
#    Recording-level splitting is impossible with one recording, so `build`
#    stops with an error.
printf '\n\033[1;33m# Recording-level splitting is impossible with one recording.\033[0m\n'
printf '\033[1;33m# iqforge does NOT fall back to window-level splitting - it stops:\033[0m\n'
run 3 build examples/bpsk_01.sigmf-meta -o "$SINGLE_DIR" || true

rm -rf "$SINGLE_DIR"
