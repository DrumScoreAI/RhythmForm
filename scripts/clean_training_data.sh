#!/bin/bash

# This script removes all .musicxml, .pdf, and .png files from the
# training_data directory, while preserving the directory structure.

# Get the directory of the current script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TRAINING_DATA_DIR="$SCRIPT_DIR/../training_data"

# Check if the training_data directory exists
if [ ! -d "$TRAINING_DATA_DIR" ]; then
    echo "Error: Directory '$TRAINING_DATA_DIR' not found."
    exit 1
fi

echo "Cleaning files from $TRAINING_DATA_DIR..."

# Find and delete the specified file types
find "$TRAINING_DATA_DIR" -type f \( -name "*.xml" -o -name "*.pdf" -o -name "*.png" \) -exec rm -vf {} \;

echo "Cleanup complete."