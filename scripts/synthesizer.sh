#!/bin/bash

# Default values
num_scores=30
num_cores=1
use_stdout=false
continuation=false

usage() {
    echo "Usage: $0 [-s|--scores NUM] [-n|--num-cores NUM] [-S|--use_stdout] [-c|--continuation]"
    echo "  -s, --scores NUM         Number of scores to generate (default: 30)"
    echo "  -n, --num-cores NUM      Number of cores to use (default: 1)"
    echo "  -S, --use_stdout         Log to stdout instead of file"
    echo "  -c, --continuation       Continue previous synthesis run"
    echo "  -h, --help               Show this help message"
}

# Parse command-line options
while [ "$1" != "" ]; do
    case $1 in
        -s | --scores)
            shift
            num_scores=$1
            ;;
        -n | --num-cores)
            shift
            num_cores=$1
            ;;
        -S | --use_stdout)
            use_stdout=true
            ;;
        -c | --continuation)
            continuation=true
            ;;
        -h | --help)
            usage
            exit
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# Validate arguments
if ! [[ "$num_cores" =~ ^[0-9]+$ ]]; then
    echo "Error: Number of cores must be an integer."
    exit 1
fi

if ! [[ "$num_scores" =~ ^[0-9]+$ ]]; then
    echo "Error: Number of scores must be an integer."
    exit 1
fi

if [ "$num_scores" -lt 1 ]; then
    echo "Error: Number of scores must be at least 1."
    exit 1
fi

if [ "$num_cores" -lt 1 ]; then
    echo "Error: Number of cores must be at least 1."
    exit 1
fi

if [ -z "$RHYTHMFORMHOME" ]; then
    echo "Error: RHYTHMFORMHOME environment variable is not set."
    exit 1
fi

if [ "$num_cores" -ge 2 ]; then
    half_cores=$((num_cores / 2))
else
    half_cores=1
fi


TRAINING_DATA_DIR="$RHYTHMFORMHOME/training_data"

mkdir -p "$TRAINING_DATA_DIR/musicxml"
mkdir -p "$TRAINING_DATA_DIR/pdfs"
mkdir -p "$TRAINING_DATA_DIR/images"
mkdir -p "$TRAINING_DATA_DIR/logs"

errfile="$TRAINING_DATA_DIR/logs/synth_log_$(date +%Y%m%d_%H%M%S).err"

if [ "$use_stdout" == "true" ]; then
    echo "Logging to stdout as requested."
    exec 2> "$errfile"
else
    logfile="$TRAINING_DATA_DIR/logs/synth_log_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to file: $logfile"
    exec > "$logfile" 2> "$errfile"
fi
echo ""
echo "------RhythmForm Score Synthesizer------"
echo "----------------------------------------"
echo "Starting data synthesis with the following parameters:"
echo "  RHYTHMFORMHOME: $RHYTHMFORMHOME"
echo "  Training data directory: $TRAINING_DATA_DIR"
echo "  Number of scores to generate: $num_scores"
echo "  Number of cores to use: $num_cores"
echo "  Continuation mode: $continuation"
echo "  Output mode: $( [ "$use_stdout" == "true" ] && echo "stdout" || echo "$logfile" )"
echo "----------------------------------------"
echo ""

clean_training_data() {
    # This function removes all .musicxml, .pdf, and .png files from the
    # training_data directory, while preserving the directory structure.
    echo "Cleaning files from $TRAINING_DATA_DIR..."

    # Find and delete the specified file types
    find "$TRAINING_DATA_DIR" -type f \( -name "*.xml" -o -name "*.pdf" -o -name "*.png" -o -name "*.json" -o -name "*.csv" \) -exec rm -vf {} \;

    echo "Cleanup complete."
}

# Call the function to clean training data
if [ "$continuation" == "false" ]; then
    echo "Starting fresh synthesis run. Cleaning training data..."
    clean_training_data
else
    echo "Continuation mode selected. Skipping cleanup of training data."
    # Determine the number of existing scores to set the start index
    existing_scores=$(find $TRAINING_DATA_DIR/musicxml -name \*\[0-9\].xml | grep -v altered 2>/dev/null | wc -l)
    if [ -z "$existing_scores" ]; then
        existing_scores=0
    fi
    echo "Existing scores found: $existing_scores"
fi

temp1=$RANDOM.tmp
temp2=$RANDOM.tmp
temp3=$RANDOM.tmp
find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered | sort -V > $temp1

# Generate synthetic scores
echo "Generating synthetic scores using $num_cores cores..."
if [ "$continuation" == "true" ] && [ "$existing_scores" -gt 0 ]; then
    echo "Continuation mode enabled."
    python $RHYTHMFORMHOME/scripts/generate_synthetic_scores.py "$num_scores" --cores "$num_cores" --continuation
else
    python $RHYTHMFORMHOME/scripts/generate_synthetic_scores.py "$num_scores" --cores "$num_cores"
fi
echo "Synthetic score generation complete."

if [ "$continuation" == "true" ] && [ "$existing_scores" -gt 0 ]; then
    find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered | sort -V > $temp2
    comm -13 $temp1 $temp2 > $temp3
    # rm -f $temp1 $temp2
else
    find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered | sort -V> $temp3
fi

this_total=$(cat $temp3 | wc -l)
start_count=$(find $TRAINING_DATA_DIR/pdfs -name "*.pdf" 2>/dev/null | wc -l)
# Convert MusicXML to PDF
echo "Converting MusicXML files to PDF using $num_cores cores..."
# Use xvfb-run -a to start a single Xvfb instance for all parallel conversions
xvfb-run -a bash -c "cat $temp3 | xargs -P $half_cores -I {} $RHYTHMFORMHOME/scripts/mscore_convert.sh {}" &
pid=$!

# Monitor progress
echo "Monitoring PDF conversion progress (PID: $pid)..."
while ps -p $pid > /dev/null; do
    current_count=$(find $TRAINING_DATA_DIR/pdfs -name "*.pdf" 2>/dev/null | wc -l)
    # Calculate how many new PDFs have been created by this run
    newly_created=$((current_count - start_count))
    echo -ne "  -> Generated $newly_created / $this_total PDFs...\r"
    if [ "$newly_created" -ge "$this_total" ]; then
        break
    fi
    sleep 2
done

# Final count and newline
current_count=$(find $TRAINING_DATA_DIR/pdfs -name "*.pdf" 2>/dev/null | wc -l)
newly_created=$((current_count - start_count))
echo -e "\n  -> Generated $newly_created / $this_total PDFs."

echo "Conversion to PDF complete."

# Create manifest file
do_or_mi="do"
# n_or_p = n for new, p for processed
echo "Creating manifest file..."
if [ "$continuation" == "false" ] || [ "$existing_scores" -eq 0 ]; then
    echo "pdf,image,musicxml,do_or_mi,n_or_p" > $TRAINING_DATA_DIR/training_data.csv
fi

echo "Populating manifest file..."
python $RHYTHMFORMHOME/scripts/build_manifest.py $TRAINING_DATA_DIR

rm -f $temp1 $temp2 $temp3
echo "Manifest file created at $TRAINING_DATA_DIR/training_data.csv."

# Prepare data for training
echo "Preparing data for training using $num_cores cores..."
python $RHYTHMFORMHOME/scripts/prepare_dataset.py --cores $num_cores
echo "Data preparation complete."

# Run tokenizer
echo "Running tokenizer (serial)..."
cd $RHYTHMFORMHOME/scripts/
python -m omr_model.tokenizer --cores $num_cores
cd -
echo "Tokenizer run complete."

# CHMOD training data
echo "Setting permissions for training data using $num_cores cores..."
find $TRAINING_DATA_DIR -type f -print0 | xargs -P "$num_cores" -0 -I {} chmod 666 {} >/dev/null 2>&1
find $TRAINING_DATA_DIR -type d -print0 | xargs -P "$num_cores" -0 -I {} chmod 777 {} >/dev/null 2>&1
echo "Permissions set."

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

echo "All data synthesis tasks completed successfully."
