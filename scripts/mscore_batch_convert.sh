#!/bin/bash
# Script to batch convert MusicXML files to PDF using MuseScore
if [ -z "$SFHOME" ]; then
  echo "Error: SFHOME environment variable is not set."
  exit 1
fi
if [ ! -d "$SFHOME/training_data/musicxml" ]; then
  echo "Error: Directory $SFHOME/training_data/musicxml does not exist."
  exit 1
fi

# Create the pdfs directory if it doesn't exist
mkdir -p $SFHOME/training_data/pdfs

# Loop through all xml files and convert them to PDF in the parallel pdfs directory
echo "Converting XML files to PDF..."
for file in $SFHOME/training_data/musicxml/*.xml; do
  filename="${file##*/}"
  mscore3 -o "$SFHOME/training_data/pdfs/${filename%.xml}.pdf" -r 300 "$file"
done
echo "Conversion complete."