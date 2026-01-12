#!/bin/bash
# Script to batch convert MusicXML files to PNG using MuseScore

# Check for input and output directories
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory $INPUT_DIR does not exist."
  exit 1
fi

# Check for MuseScore path
if [ -z "$MUSESCORE_PATH" ]; then
  echo "Error: MUSESCORE_PATH environment variable is not set."
  echo "Please set it to the path of your MuseScore executable."
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all xml files and convert them to PNG
echo "Converting MusicXML files to PNG in $OUTPUT_DIR..."
for file in "$INPUT_DIR"/*.xml; do
  if [ -f "$file" ]; then
    filename=$(basename -- "$file")
    output_file="$OUTPUT_DIR/${filename%.xml}.png"
    echo "Converting $file to $output_file"
    "$MUSESCORE_PATH" -o "$output_file" "$file"
  fi
done

echo "Conversion complete."