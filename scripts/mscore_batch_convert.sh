#!/bin/bash
# Script to batch convert MusicXML files to PDF using MuseScore
if [ -z "$RHYTHMFORMHOME" ]; then
  echo "Error: RHYTHMFORMHOME environment variable is not set."
  exit 1
fi

$TRAINING_DATA_DIR="$RHYTHMFORMHOME/training_data"

if [ ! -d "$TRAINING_DATA_DIR/musicxml" ]; then
  echo "Error: Directory $TRAINING_DATA_DIR/musicxml does not exist."
  exit 1
fi

# Create the pdfs directory if it doesn't exist
mkdir -p $TRAINING_DATA_DIR/pdfs

# Loop through all xml files and convert them to PDF in the parallel pdfs directory
echo "Converting XML files to PDF..."
for file in $TRAINING_DATA_DIR/musicxml/*.xml; do
  filename="${file##*/}"
  $MUSESCORE_PATH -o "$TRAINING_DATA_DIR/pdfs/${filename%.xml}.pdf" -r 300 "$file"
done
echo "Conversion complete."