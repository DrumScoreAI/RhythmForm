#!/bin/bash
# Default values

TRAINING_DATA_DIR="$RHYTHMFORMHOME/training_data"
cd $RHYTHMFORMHOME/scripts || exit 1

num_cores=$1

echo "Manifest file created at $TRAINING_DATA_DIR/training_data.csv."

# Prepare data for training
echo "Preparing data for training using $num_cores cores..."
python $RHYTHMFORMHOME/scripts/prepare_dataset.py --cores "$num_cores"
echo "Data preparation complete."

# Run tokenizer
echo "Running tokenizer (serial)..."
cd $RHYTHMFORMHOME/scripts/
python -m omr_model.tokenizer --cores $num_cores
cd -
echo "Tokenizer run complete."

# CHMOD training data
echo "Setting permissions for training data using $num_cores cores..."
find $TRAINING_DATA_DIR -type f -print0 | xargs -P "$num_cores" -0 -I {} chmod 666 {}
find $TRAINING_DATA_DIR -type d -print0 | xargs -P "$num_cores" -0 -I {} chmod 777 {}
echo "Permissions set."

echo "All data synthesis tasks completed successfully."

# ZIP and upload datasets to S3 (if configured)
echo "Uploading datasets to S3..."
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$S3_ENDPOINT_URL" ]; then
    echo "AWS credentials not set. Skipping upload."
    exit 0
else
    echo "AWS credentials found. Proceeding with upload."
fi
python $RHYTHMFORMHOME/scripts/zip_and_upload_dataset.py --note "$num_scores scores synthesized on $(date +"%Y-%m-%d %T")" --bucket-name "rhythmformdatasets"
echo "Zip and upload complete."