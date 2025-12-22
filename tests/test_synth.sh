#!/bin/bash

if [ -z "$RHYTHMFORMHOME" ]; then
    RHYTHMFORMHOME=$1
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
        echo "$file -- EXISTS"
    fi
done
echo "Required files missing: $missing"
tokens=$(head -n -1 $TRAINING_DATA_DIR/tokenizer_vocab.json | tail -n +2 | wc -l)
echo "Number of tokens in tokenizer_vocab.json: $tokens"
if [ $tokens -le 4 ] || [ $missing -ne 0 ]; then
    echo "Some required files are missing or tokenizer_vocab.json has insufficient tokens."
    exit 1
else
    echo "All required files are present."
    exit 0
fi