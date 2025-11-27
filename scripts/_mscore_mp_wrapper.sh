#!/bin/bash
# Script to batch convert MusicXML files to PDF using MuseScore
# For use as a wrapper called by synthesizer.sh with xargs for parallel processing

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
echo $filename $xml_dir $pdf_dir ${pdf_dir}/${filename%.xml}.pdf
xvfb-run -a $MUSESCORE_PATH -o "${pdf_dir}/${filename%.xml}.pdf" -r 300 "${xml_file}" 2>/dev/null