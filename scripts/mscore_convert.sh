#!/bin/bash
# Script to convert MusicXML to PDF using MuseScore
# Simplified wrapper for use with single Xvfb instance

if [ $# -ne 1 ]; then
  echo "Usage: $0 <abs_path_to_musicxml_file>"
  exit 1
fi

if [ -z "$MUSESCORE_PATH" ]; then
  echo "Error: MUSESCORE_PATH environment variable is not set."
  exit 1
fi

xml_file=$1

filename="${xml_file##*/}"
xml_dir="${xml_file%/*}"
pdf_dir="${xml_dir}/../pdfs"

# Run MuseScore
"$MUSESCORE_PATH" -o "${pdf_dir}/${filename%.xml}.pdf" -r 300 "${xml_file}" 2>&1 | grep -v -E "pw.context|pw.conf|libOpenGL|libjack|libnss3|libpipewire"
