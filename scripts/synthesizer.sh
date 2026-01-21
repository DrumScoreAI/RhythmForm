#!/bin/bash
set -e
# Default values
num_scores=30
num_cores=1
use_stdout=false
continuation=false

usage() {
    echo "Usage: $0 [-s|--scores NUM] [-n|--num-cores NUM] [-S|--use_stdout] [-c|--continuation]"
    echo "  -s, --scores NUM         Target number of scores, s (default: 30). If continuation is set, will generate s - c scores, where c is the current number of scores."
    echo "  -n, --num-cores NUM      Number of cores to use (default: 1)"
    echo "  -S, --use_stdout         Log to stdout instead of file"
    echo "  -c, --continuation       Continue previous synthesis run"
    echo "  -w, --write_smt          Write SMT files to training_data/smt directory"
    echo "  -m, --markov-ratio       Ratio of scores to generate using the Markov model (default: 0.8)"
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
        -w | --write_smt)
            write_smt=true
            ;;
        -m | --markov-ratio)
            shift
            markov_ratio=$1
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

    # Find and delete the specified file types, excluding the fine_tuning directory
    find "$TRAINING_DATA_DIR" -path "$TRAINING_DATA_DIR/fine_tuning" -prune -o -type f \( -name "*.xml" -o -name "*.pdf" -o -name "*.png" -o -name "*.csv" -o -name "*.json" -o -name "*.smt" \) -exec rm -vf {} +

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
    start_index=$existing_scores
    echo "Existing scores found: $existing_scores"
fi

temp1=$RANDOM.tmp
temp2=$RANDOM.tmp
temp3=$RANDOM.tmp
find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered | sort -n > $temp1

# Generate synthetic scores
echo "Generating synthetic scores using $num_cores cores..."
if [ "$continuation" == "true" ] && [ "$existing_scores" -gt 0 ]; then
    echo "Continuation mode enabled."
    python $RHYTHMFORMHOME/scripts/generate_synthetic_scores.py "$num_scores" --cores "$num_cores" --markov-model "$TRAINING_DATA_DIR/markov_model.pkl" --start-index "$start_index"
else
    python $RHYTHMFORMHOME/scripts/generate_synthetic_scores.py "$num_scores" --cores "$num_cores" --markov-model "$TRAINING_DATA_DIR/markov_model.pkl"
fi
echo "Synthetic score generation complete."

if [ "$continuation" == "true" ] && [ "$existing_scores" -gt 0 ]; then
    find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered | sort -n > $temp2
    comm -13 $temp1 $temp2 > $temp3
    # rm -f $temp1 $temp2
else
    find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered | sort -n > $temp3
fi

this_total=$(cat $temp3 | wc -l)
start_count=$(find $TRAINING_DATA_DIR/pdfs -name "*.pdf" 2>/dev/null | wc -l)
# Convert MusicXML to PDF
echo "Converting MusicXML files to PDF using $half_cores cores..."
# Use xvfb-run -a to start a single Xvfb instance for all parallel conversions
# Filter out "Invalid QML element name" messages from mscore
if ! xvfb-run -a bash -c "cat $temp3 | xargs -P $half_cores -I {} bash -c '$RHYTHMFORMHOME/scripts/mscore_convert.sh -i \"\$0\" -f pdf &> >(grep -v \"Invalid QML element name\" >&2)' {}"; then
    echo "Warning: PDF conversion may have failed for some files. One or more mscore_convert.sh processes may have been terminated."
    echo "This can happen due to high memory usage. Check the logs of the failed pod for more details."
fi
# The process is now run in the foreground, so we don't need to monitor it with a while loop.
# The script will wait here until all conversions are complete.


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
if [ "$write_smt" == "true" ]; then
    echo "SMT writing enabled. SMT files will be written to training_data/smt directory."
    python $RHYTHMFORMHOME/scripts/prepare_dataset.py --cores $num_cores --write-smt
else
    python $RHYTHMFORMHOME/scripts/prepare_dataset.py --cores $num_cores
fi
if [ $? -ne 0 ]; then
    echo "Error during data preparation."
    exit 1
fi

# Add a check to ensure dataset.json was created on a fresh run
if [ "$continuation" == "false" ] && [ ! -f "$TRAINING_DATA_DIR/dataset.json" ]; then
    echo "CRITICAL ERROR: dataset.json was not created during a fresh run."
    echo "This indicates that no source MusicXML/PDF files were found or processed."
    echo "Please check the logs for errors during score generation, PDF conversion, or manifest building."
    echo "Aborting before cleanup to preserve generated files for inspection."
    exit 1
fi

echo "Data preparation complete."

# Clean up
echo "Cleaning up any orphaned files..."
bash $RHYTHMFORMHOME/scripts/clean_up_synth.sh $TRAINING_DATA_DIR $RHYTHMFORMHOME/scripts $num_cores
echo "Cleanup complete."

# Run tokenizer
echo "Running tokenizer (serial)..."
python $RHYTHMFORMHOME/scripts/omr_model/tokenizer.py --cores $num_cores
echo "Tokenizer run complete."

# CHMOD training data
# echo "Setting permissions for training data using $num_cores cores..."
# find $TRAINING_DATA_DIR -type f -print0 | xargs -P "$num_cores" -0 -I {} chmod 666 {} >/dev/null 2>&1
# find $TRAINING_DATA_DIR -type d -print0 | xargs -P "$num_cores" -0 -I {} chmod 777 {} >/dev/null 2>&1
# echo "Permissions set."

# ZIP and upload datasets to S3 (if configured)
echo "Uploading datasets to S3..."
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$S3_ENDPOINT_URL" ]; then
    echo "AWS credentials not set. Skipping upload."
    exit 0
else
    echo "AWS credentials found. Proceeding with upload."
fi
python $RHYTHMFORMHOME/scripts/zip_and_upload_dataset.py --note "$num_scores scores synthesized on $(date +"%Y-%m-%d %T")" --bucket-name "rhythmformdatasets" --tar-only --exclude-dirs "logs" "pdfs"
echo "Zip and upload complete."

echo "All data synthesis tasks completed successfully."
