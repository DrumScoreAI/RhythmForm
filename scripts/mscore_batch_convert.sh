#!/bin/bash
# Script to batch convert MusicXML files to PDF using MuseScore
if [ -z "$RHYTHMFORMHOME" ]; then
  echo "Error: RHYTHMFORMHOME environment variable is not set."
  exit 1
fi
if [ ! -d "$RHYTHMFORMHOME/training_data/musicxml" ]; then
  echo "Error: Directory $RHYTHMFORMHOME/training_data/musicxml does not exist."
  exit 1
fi

# Create the pdfs directory if it doesn't exist
mkdir -p $RHYTHMFORMHOME/training_data/pdfs

# Loop through all xml files and convert them to PDF in the parallel pdfs directory
echo "Converting XML files to PDF..."
for file in $RHYTHMFORMHOME/training_data/musicxml/*.xml; do
  filename="${file##*/}"
  $MUSESCORE_PATH -o "$RHYTHMFORMHOME/training_data/pdfs/${filename%.xml}.pdf" -r 300 "$file"
done
echo "Conversion complete."