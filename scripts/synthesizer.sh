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
    existing_scores=$(find $TRAINING_DATA_DIR/musicxml -name \*\[0-9\].xml | grep -v altered 2>/dev/null | wc -l)
    if [ -z "$existing_scores" ]; then
        existing_scores=0
    fi
    echo "Existing scores found: $existing_scores"
fi

temp1=$RANDOM.tmp
temp2=$RANDOM.tmp
temp3=$RANDOM.tmp
find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered > $temp1

# Generate synthetic scores
echo "Generating synthetic scores using $num_cores cores..."
if [ "$continuation" == "true" ] && [ "$existing_scores" -gt 0 ]; then
    echo "Continuation mode enabled."
    python generate_synthetic_scores.py "$num_scores" --cores "$num_cores" --continuation
else
    python generate_synthetic_scores.py "$num_scores" --cores "$num_cores"
fi
echo "Synthetic score generation complete."

if [ "$continuation" == "true" ] && [ "$existing_scores" -gt 0 ]; then
    find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered > $temp2
    grep -Fv -f $temp1 $temp2 > $temp3
    # rm -f $temp1 $temp2
else
    find $TRAINING_DATA_DIR/musicxml -name "*.xml" | grep -v altered > $temp3
fi

this_total=$(cat $temp3 | wc -l)
start_count=$(find $TRAINING_DATA_DIR/pdfs -name "*.pdf" 2>/dev/null | wc -l)
# Convert MusicXML to PDF
echo "Converting MusicXML files to PDF using $num_cores cores..."
cat $temp3 | xargs -P "$num_cores" -I {} $RHYTHMFORMHOME/scripts/_mscore_mp_wrapper.sh {} &
pid=$!

# Monitor progress
echo "Monitoring PDF conversion progress (PID: $pid)..."
while ps -p $pid > /dev/null; do
    current_count=$(find $TRAINING_DATA_DIR/pdfs -name "*.pdf" 2>/dev/null | wc -l)
    # Calculate how many new PDFs have been created by this run
    newly_created=$((current_count - start_count))
    echo -ne "  -> Generated $newly_created / $this_total PDFs...\r"
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
    echo "pdf,musicxml,do_or_mi,n_or_p" > $TRAINING_DATA_DIR/training_data.csv
fi

find "$TRAINING_DATA_DIR/musicxml/" -name "*[0-9].xml" | while read -r xml; do
    xml_bn=$(basename "$xml")
    pdf="${xml_bn%.xml}.pdf"
    grep -q "$xml" "$temp1" 2>/dev/null
    if [ $? -eq 0 ]; then
        sed "s/$pdf,$xml_bn,$do_or_mi,n/$pdf,$xml_bn,$do_or_mi,p/g" $TRAINING_DATA_DIR/training_data.csv > $TRAINING_DATA_DIR/training_data_tmp.csv
        # cat $TRAINING_DATA_DIR/training_data_tmp.csv
    else
        echo "$pdf,$xml_bn,$do_or_mi,n" >> $TRAINING_DATA_DIR/training_data.csv
    fi
done

rm -f $temp1 $temp2 $temp3
echo "Manifest file created at $TRAINING_DATA_DIR/training_data.csv."

# Prepare data for training
echo "Preparing data for training using $num_cores cores..."
python prepare_dataset.py --cores $num_cores
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
