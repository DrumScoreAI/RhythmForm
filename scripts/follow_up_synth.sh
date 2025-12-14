#!/bin/bash
# Default values

TRAINING_DATA_DIR="$RHYTHMFORMHOME/training_data"
cd $RHYTHMFORMHOME/scripts || exit 1

num_cores=$1

echo "Manifest file created at $TRAINING_DATA_DIR/training_data.csv."

# Prepare data for training
echo "Preparing data for training using $num_cores cores..."
python prepare_dataset.py --cores "$num_cores"
echo "Data preparation complete."

# Run tokenizer
echo "Running tokenizer (serial)..."
python -m omr_model.tokenizer --cores $num_cores
echo "Tokenizer run complete."

# CHMOD training data
echo "Setting permissions for training data using $num_cores cores..."
find $TRAINING_DATA_DIR -type f -print0 | xargs -P "$num_cores" -0 -I {} chmod 666 {}
find $TRAINING_DATA_DIR -type d -print0 | xargs -P "$num_cores" -0 -I {} chmod 777 {}
echo "Permissions set."

echo "All data synthesis tasks completed successfully."