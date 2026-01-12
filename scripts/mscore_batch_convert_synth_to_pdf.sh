#!/bin/bash
# Script to batch convert synthetic MusicXML files to PDF using MuseScore

# Check for MUSESCORE_PATH
if [ -z "$MUSESCORE_PATH" ]; then
  echo "Error: MUSESCORE_PATH environment variable is not set."
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory $INPUT_DIR does not exist."
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Output directory $OUTPUT_DIR does not exist."
  exit 1
fi

echo "Converting synthetic MusicXML files to PDF in $OUTPUT_DIR..."
for file in "$INPUT_DIR"/synthetic_score_*.xml; do
  if [ -f "$file" ]; then
    filename=$(basename -- "$file")
    output_file="$OUTPUT_DIR/${filename%.xml}.pdf"
    echo "Converting $file to $output_file"
    "$MUSESCORE_PATH" -o "$output_file" "$file"
  fi
done
echo "Conversion complete."
