#!/bin/bash
# Script to convert MusicScore files to other formats using MuseScore

usage() {
  echo "Usage: $0 -i <input_file> [-f <format>] [-o <output_path>]"
  echo "  -i <input_file>   Path to the input MusicScore (.mscz) or MusicXML (.xml, .musicxml) file."
  echo "  -f <format>       Output format. One of 'pdf', 'musicxml'. Default is 'pdf'."
  echo "  -o <output_path>  Path for the output file. If not provided, output is placed in a format-specific directory (e.g., '../pdfs', '../musicxml') relative to the input file's location."
  exit 1
}

output_format="pdf"
output_path=""
input_file=""

while getopts "i:f:o:h" opt; do
  case $opt in
    i) input_file="$OPTARG" ;;
    f) output_format="$OPTARG" ;;
    o) output_path="$OPTARG" ;;
    h) usage ;;
    \?) usage ;;
  esac
done

if [ -z "$input_file" ]; then
  echo "Error: Input file not specified."
  usage
fi

if [ ! -f "$input_file" ]; then
    echo "Error: Input file not found at '$input_file'"
    usage
fi

if [ -z "$MUSESCORE_PATH" ]; then
  echo "Error: MUSESCORE_PATH environment variable is not set."
  exit 1
fi

if [ "$output_format" != "pdf" ] && [ "$output_format" != "musicxml" ]; then
    echo "Error: Invalid output format '$output_format'. Must be 'pdf' or 'musicxml'."
    usage
fi

input_dir=$(dirname "$input_file")
filename=$(basename "$input_file")
base_filename="${filename%.*}"

# Sanitize filename for output: lowercase, no spaces, and remove special characters
sanitized_basename=$(echo "$base_filename" | tr '[:upper:]' '[:lower:]' | tr -d ' ' | sed 's/[^a-z0-9._-]//g')

# Determine output path if not explicitly provided
if [ -z "$output_path" ]; then
    if [ "$output_format" = "pdf" ]; then
        output_dir_name="pdfs"
    else # musicxml
        output_dir_name="musicxml"
    fi
    
    # Place the output directory alongside the input file's directory
    output_dir="$input_dir/../$output_dir_name"
    
    mkdir -p "$output_dir"
    output_extension=$output_format
    output_path="${output_dir}/${sanitized_basename}.${output_extension}"
fi

echo "Converting '$input_file' to '$output_path'..."

# Run MuseScore, filtering out common noise
"$MUSESCORE_PATH" -o "${output_path}" "${input_file}" 2>&1 | grep -v -E "pw.context|pw.conf|libOpenGL|libjack|libnss3|libpipewire|ALSA|Invalid QML element name"

# Check the exit status of the MuseScore command
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Conversion successful."
else
    echo "Error during conversion."
fi

