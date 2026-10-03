#!/usr/bin/env bash
# Shrink TTS audio ~10-20x for weak networks (needs ffmpeg). Usage: to_opus.sh in.wav out.ogg
ffmpeg -y -i "$1" -c:a libopus -b:a 16k -ac 1 -application voip "$2"
