#!/bin/bash

TRAINING_DATA_DIR=$1
SCRIPTS_DIR=$2
CORES=$3

cd $TRAINING_DATA_DIR

find musicxml/ -print0 -name *[0-9].xml | xargs -0 -P $CORES -n 1 -I {} $SCRIPTS_DIR/_clean_up_synth.sh $TRAINING_DATA_DIR {} >/dev/null &
pid=$!

# Monitor progress
echo "Monitoring deletion of unpaired files (PID: $pid)..."
start_count_xml=$(find musicxml -type f 2>/dev/null | wc -l)
start_count_pdf=$(find pdfs -type f 2>/dev/null | wc -l)
start_count_image=$(find images -type f 2>/dev/null | wc -l)
while ps -p $pid > /dev/null; do
    current_count_xml=$(find musicxml -type f 2>/dev/null | wc -l)
    current_count_pdf=$(find pdfs -type f 2>/dev/null | wc -l)
    current_count_image=$(find images -type f 2>/dev/null | wc -l)
    # Calculate how many new PDFs have been created by this run
    deleted=$((start_count - current_count))
    echo -ne "  -> Removed $deleted unpaired files...\r"
    sleep 2
done



for i in `grep image_path dataset.json | awk -F\" '{print $4}'`; do exists=`ls $i`; if [ -z $exists ]; then echo $i does not exist; fi; done