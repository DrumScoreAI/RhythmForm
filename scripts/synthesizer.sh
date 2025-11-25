#!/bin/bash

if [ "$#" -eq 1 ]; then
    num_scores=$1
fi

if [ -z "$num_scores" ]; then
    num_scores=30
fi

if ! [[ "$num_scores" =~ ^[0-9]+$ ]]; then
    echo "Error: Number of scores must be an integer."
    exit 1
fi

if [ "$num_scores" -lt 1 ]; then
    echo "Error: Number of scores must be at least 1."
    exit 1
fi

if [ -z "$RHYTHMFORMHOME" ]; then
    echo "Error: RHYTHMFORMHOME environment variable is not set."
    exit 1
fi

clean_training_data() {
    # This function removes all .musicxml, .pdf, and .png files from the
    # training_data directory, while preserving the directory structure.

    # Get the directory of the current script
    TRAINING_DATA_DIR="$RHYTHMFORMHOME/training_data"

    mkdir -p "$TRAINING_DATA_DIR/musicxml"
    mkdir -p "$TRAINING_DATA_DIR/pdfs"
    mkdir -p "$TRAINING_DATA_DIR/images"

    echo "Cleaning files from $TRAINING_DATA_DIR..."

    # Find and delete the specified file types
    find "$TRAINING_DATA_DIR" -type f \( -name "*.xml" -o -name "*.pdf" -o -name "*.png" -o -name "*.json" -o -name "*.csv" \) -exec rm -vf {} \;

    echo "Cleanup complete."
}

# Call the function to clean training data
clean_training_data

# Generate synthetic scores
echo "Generating synthetic scores..."
python generate_synthetic_scores.py "$num_scores"
echo "Synthetic score generation complete."

# Convert MusicXML to PDF
echo "Converting MusicXML files to PDF..."
$RHYTHMFORMHOME/scripts/mscore_batch_convert.sh 2>/dev/null
echo "Conversion to PDF complete."

# Create manifest file
echo "Creating manifest file..."
echo "pdf,musicxml,do_or_mi" > $TRAINING_DATA_DIR/training_data.csv
do_or_mi="do"
for xml in `ls $TRAINING_DATA_DIR/musicxml/*.xml`; do
    xml_bn=$(basename "$xml")
    pdf="${xml_bn%.xml}.pdf"
    echo "$pdf,$xml_bn,$do_or_mi" >> $TRAINING_DATA_DIR/training_data.csv
done

# Prepare data for training
echo "Preparing data for training..."
python prepare_dataset.py
echo "Data preparation complete."

# Run tokenizer
echo "Running tokenizer..."
python -m omr_model.tokenizer
echo "Tokenizer run complete."

# CHMOD training data
echo "Setting permissions for training data..."
find $TRAINING_DATA_DIR -type f -exec chmod 666 {} \;
find $TRAINING_DATA_DIR -type d -exec chmod 777 {} \;
echo "Permissions set."

echo "All data synthesis tasks completed successfully."
echo "You are ready to train the OMR model."