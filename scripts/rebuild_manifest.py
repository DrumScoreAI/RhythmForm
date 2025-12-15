import os
import csv
from pathlib import Path
import sys

def rebuild_manifest(training_data_dir: Path|str):
    """
    Rebuilds the training data manifest (training_data.csv) by checking for
    the presence of corresponding PDF files for each MusicXML file.

    This script is a more performant Python replacement for the original
    rebuild_manifest.sh shell script.
    """
    try:
        # Assume the script is run from the project root, where the 'training_data' directory exists.
        if training_data_dir is None:
            sys.exit("Error: training_data_dir must be provided.")
        elif isinstance(training_data_dir, str):
            training_data_dir = Path(training_data_dir)
        pdfs_dir = training_data_dir / 'pdfs'
        musicxml_dir = training_data_dir / 'musicxml'
        manifest_path = training_data_dir / 'training_data.csv'

        if not all([training_data_dir.is_dir(), pdfs_dir.is_dir(), musicxml_dir.is_dir(), manifest_path.is_file()]):
            print("Error: Required directories or files not found. Make sure you are running this script from the project root directory.", file=sys.stderr)
            sys.exit(1)

        print("Scanning for PDF files...")
        # Create a set of PDF basenames for efficient lookup
        pdf_filenames = {f.name for f in pdfs_dir.glob('*.pdf')}
        print(f"Found {len(pdf_filenames)} PDF files.")

        print("Updating manifest (training_data.csv)...")
        
        # Read the manifest into memory
        with open(manifest_path, 'r', newline='') as f:
            reader = csv.reader(f)
            manifest_data = list(reader)

        updated_rows = 0
        # Iterate through the manifest and update rows
        for i, row in enumerate(manifest_data):
            if len(row) == 4:
                pdf_fn, xml_fn, status, processing_status = row
                # Check if the corresponding PDF exists and the status needs updating
                if processing_status == 'n' and pdf_fn in pdf_filenames:
                    manifest_data[i][3] = 'p' # Update status to 'processed'
                    updated_rows += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1} rows...", end='\r')

        # Write the updated data back to the manifest
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(manifest_data)

        print(f"\nManifest update complete. Updated {updated_rows} rows.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you are running the script from the project's root directory.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        rebuild_manifest(sys.argv[1])
    else:
        if (Path(os.environ['RHYTHMFORMHOME']) / 'training_data').exists():
            rebuild_manifest(Path(os.environ['RHYTHMFORMHOME']) / 'training_data')
        else:
            sys.exit("Error: training_data_dir must be provided as an argument or be a subdirectory to RHYTHMFORMHOME environment variable.")
            
