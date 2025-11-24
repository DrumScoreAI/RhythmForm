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

    # Check if the training_data directory exists
    if [ ! -d "$TRAINING_DATA_DIR" ]; then
        echo "Error: Directory '$TRAINING_DATA_DIR' not found."
        exit 1
    fi

    echo "Cleaning files from $TRAINING_DATA_DIR..."

    # Find and delete the specified file types
    find "$TRAINING_DATA_DIR" -type f \( -name "*.xml" -o -name "*.pdf" -o -name "*.png" \) -exec rm -vf {} \;

    echo "Cleanup complete."
}

# Call the function to clean training data
clean_training_data

# Generate synthetic scores
echo "Generating synthetic scores..."
python generate_synthetic_scores.py --num_scores "$num_scores"
echo "Synthetic score generation complete."

# Create manifest file
echo "Creating manifest file..."
echo "pdf,musicxml,do_or_mi" > $TRAINING_DATA_DIR/training_data.csv
do_or_mi="do"
for pdf in `ls $TRAINING_DATA_DIR/pdfs/*.pdf`; do
    pdf_bn=$(basename "$pdf")
    musicxml="${pdf_bn%.pdf}.xml"
    echo "$pdf,$musicxml,$do_or_mi" >> $TRAINING_DATA_DIR/training_data.csv
done

# Prepare data for training
echo "Preparing data for training..."
python prepare_dataset.py
echo "Data preparation complete."

# Run tokenizer
echo "Running tokenizer..."
python -m omr_model.tokenizer
echo "Tokenizer run complete."

echo "All data synthesis tasks completed successfully."
echo "You are ready to train the OMR model."