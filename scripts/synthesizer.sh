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
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--scores)
            num_scores="$2"
            shift; shift
            ;;
        -n|--num-cores)
            num_cores="$2"
            shift; shift
            ;;
        -S|--use_stdout)
            use_stdout=true
            shift
            ;;
        -c|--continuation)
            continuation=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
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
    existing_scores=$(ls $TRAINING_DATA_DIR/musicxml/*[0-9].xml 2>/dev/null | wc -l)
    if [ -z "$existing_scores" ]; then
        existing_scores=0
    fi
    echo "Existing scores found: $existing_scores"
fi

# Generate synthetic scores
echo "Generating synthetic scores using $num_cores cores..."
if [ "$continuation" == "true" ] && [ "$existing_scores" -gt 0 ]; then
    echo "Continuation mode enabled."
    python generate_synthetic_scores.py "$num_scores" --cores "$num_cores" --continuation
else
    python generate_synthetic_scores.py "$num_scores" --cores "$num_cores"
fi
echo "Synthetic score generation complete."

# Convert MusicXML to PDF
echo "Converting MusicXML files to PDF using $num_cores cores..."
find $TRAINING_DATA_DIR/musicxml -name "*.xml" -print0 | xargs -0 -P "$num_cores" -I {} $RHYTHMFORMHOME/scripts/_mscore_mp_wrapper.sh {}
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
echo "Preparing data for training using $num_cores cores..."
python prepare_dataset.py --cores "$num_cores"
echo "Data preparation complete."

# Run tokenizer
echo "Running tokenizer (serial)..."
python -m omr_model.tokenizer
echo "Tokenizer run complete."

# CHMOD training data
echo "Setting permissions for training data using $num_cores cores..."
find $TRAINING_DATA_DIR -type f -print0 | xargs -P "$num_cores" -0 -I {} chmod 666 {}
find $TRAINING_DATA_DIR -type d -print0 | xargs -P "$num_cores" -0 -I {} chmod 777 {}
echo "Permissions set."

echo "All data synthesis tasks completed successfully."
echo "You are ready to train the OMR model."