#!/bin/bash

if [ -z "$RHYTHMFORMHOME" ]; then
    echo "RHYTHMFORMHOME is not set. Please set it to the RhythmForm home directory."
    exit 1
fi

TRAINING_DATA_DIR="$RHYTHMFORMHOME/training_data"

required_files=(
    "dataset.json"
    "tokenizer_vocab.json"
    "training_data.csv"
    "images/synthetic_score_1_level_0.png"
    "musicxml/synthetic_score_1_level_0.xml"
    "pdfs/synthetic_score_1_level_0.pdf"
)
missing=0
for file in "${required_files[@]}"; do
    if [ ! -f "$TRAINING_DATA_DIR/$file" ]; then
        echo "$file -- MISSING"
        missing=$((missing + 1))
    else
        echo "$file -- OK"
    fi
done
if [ $missing -ne 0 ]; then
    exit 1
fi