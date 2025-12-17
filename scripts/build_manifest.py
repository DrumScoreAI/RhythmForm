import os
import csv
from pathlib import Path
import sys
import json
import pandas as pd

def find_dataset_entries(row):
    if row['is_in_pdf'] and row['is_in_musicxml'] and row['is_in_image']:
        return 'p'
    elif row['is_in_pdf'] and row['is_in_musicxml'] and not row['is_in_image']:
        return 'n'
    else:
        return 'unknown'

def rebuild_manifest(training_data_dir: Path|str):
    """
    Rebuilds the training data manifest (training_data.csv) by checking for
    the presence of corresponding PDF files for each MusicXML file. If a MusicXML
    file is not in the manifest, it is added.

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
        images_dir = training_data_dir / 'images'
        musicxml_dir = training_data_dir / 'musicxml'
        manifest_path = training_data_dir / 'training_data.csv'
        dataset_file = training_data_dir / 'dataset.json'

        if not all([training_data_dir.is_dir(), pdfs_dir.is_dir(), images_dir.is_dir(), musicxml_dir.is_dir(), manifest_path.is_file()]):
            print("Error: Required directories or files not found. Make sure you are running this script from the project root directory.", file=sys.stderr)
            sys.exit(1)
        
        print("Scanning for MusicXML files...")
        # Create a set of MusicXML basenames for efficient lookup
        musicxml_filenames = pd.DataFrame([f.stem for f in musicxml_dir.glob('*[0-9].xml')], columns=['stem']).sort_values(by='stem')
        print(f"Found {len(musicxml_filenames)} MusicXML files.")

        print("Scanning for PDF files...")
        # Create a set of PDF basenames for efficient lookup
        pdf_filenames = pd.DataFrame([f.stem for f in pdfs_dir.glob('*.pdf')], columns=['stem']).sort_values(by='stem')
        print(f"Found {len(pdf_filenames)} PDF files.")

        print("Scanning for Image files...")
        # Create a set of Image basenames for efficient lookup
        image_filenames = pd.DataFrame([f.stem for f in images_dir.glob('*.png')], columns=['stem']).sort_values(by='stem')
        print(f"Found {len(image_filenames)} Image files.")

        print("Loading dataset.json if present...")
        dataset = {}
        if dataset_file.is_file():
            with open(dataset_file, 'r') as df:
                dataset = json.load(df)
            print(f"Loaded dataset.json with {len(dataset)} entries.")
        else:
            print("No dataset.json found, proceeding without it.")
        
        dataset = pd.DataFrame.from_dict(dataset)

        print("Updating manifest (training_data.csv)...")
        
        # Read the manifest into memory and create a lookup for xml filenames
        manifest_data = pd.read_csv(manifest_path)

        updated_rows = 0
        added_rows = 0

        manifest_data['stem'] = manifest_data['musicxml'].apply(lambda x: Path(x).stem)
        manifest_data.sort_values(by='stem', inplace=True)

        # If anything is missing, assume it needs processed
        manifest_data['n_or_p'] = manifest_data.apply(lambda row: 'n' if row['pdf'] == '' or row['musicxml'] == '' or row['image'] == '' else 'unknown', axis=1)

        pdf_filenames['in_manifest'] = pdf_filenames['stem'].isin(manifest_data['pdf'])
        musicxml_filenames['in_manifest'] = musicxml_filenames['stem'].isin(manifest_data['musicxml'])
        image_filenames['in_manifest'] = image_filenames['stem'].isin(manifest_data['image'])
        
        # update manifest with cases where there are pdfs, musicxml and images not listed
        unlisted_pdfs = pdf_filenames[~pdf_filenames['in_manifest']]
        unlisted_musicxmls = musicxml_filenames[~musicxml_filenames['in_manifest']]
        unlisted_images = image_filenames[~image_filenames['in_manifest']]

        # We need at least MusicXML and PDF to form a valid entry. Image might be missing (needs generation).
        if unlisted_musicxmls.empty or unlisted_pdfs.empty:
            print("No unlisted MusicXML or PDF files found to add to manifest.")
        else:
            # Find stems present in both MusicXML and PDF
            new_stems = unlisted_musicxmls.merge(unlisted_pdfs, on='stem')
            
            # If we have images, we can merge them in to check existence, but we shouldn't require them.
            # Let's just iterate over the found stems.
            for _, row in new_stems.iterrows():
                stem = row['stem']
                pdf_fn = f"{stem}.pdf"
                xml_bn = f"{stem}.xml"
                image_fn = f"{stem}.png"
                
                do_or_mi = 'do'  # Default to 'do' for new entries
                # We set it to 'n' initially, but the recalculation later will fix it based on actual file existence
                processing_status = 'n' 
                
                new_row = {'pdf': pdf_fn, 'musicxml': xml_bn, 'image': image_fn, 'do_or_mi': do_or_mi, 'n_or_p': processing_status}
                manifest_data = pd.concat([manifest_data, pd.DataFrame([new_row])], ignore_index=True)
                added_rows += 1
            print(f"Added {added_rows} new rows to manifest.")

        manifest_data['is_in_pdf'] = manifest_data['pdf'].isin(pdf_filenames['stem'])
        manifest_data['is_in_musicxml'] = manifest_data['musicxml'].isin(musicxml_filenames['stem'])
        manifest_data['is_in_image'] = manifest_data['image'].isin(image_filenames['stem'])

        manifest_data['n_or_p'] = manifest_data.apply(find_dataset_entries, axis=1)

        # Write the updated data back to the manifest
        manifest_data.drop(columns=['stem', 'is_in_pdf', 'is_in_musicxml', 'is_in_image'], inplace=True)
        manifest_data.to_csv(manifest_path, index=False)

        print(f"\nManifest update complete. Updated {updated_rows} rows, added {added_rows} new rows.")

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
            
