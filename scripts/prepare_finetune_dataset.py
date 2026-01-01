import os
from pathlib import Path
import music21
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from tqdm import tqdm
import argparse
from scripts.omr_model import config
from scripts.omr_model.utils import musicxml_to_smt

# --- Configuration ---
IMAGE_DPI = config.IMAGE_DPI
PROJECT_ROOT = config.PROJECT_ROOT
TRAINING_DATA_DIR = config.TRAINING_DATA_DIR

# --- Fine-tuning specific paths ---
FINE_TUNE_DIR = TRAINING_DATA_DIR / 'fine_tuning'
PDF_INPUT_DIR = FINE_TUNE_DIR / 'pdfs'
XML_INPUT_DIR = FINE_TUNE_DIR / 'musicxml'
IMAGE_OUTPUT_DIR = FINE_TUNE_DIR / 'images'
SMT_OUTPUT_DIR = FINE_TUNE_DIR / 'smt' # Added for clarity
DATASET_JSON_PATH = FINE_TUNE_DIR / 'finetune_dataset.json'

def process_pdf(pdf_path):
    """
    Processes a single PDF file for the fine-tuning dataset:
    1. Finds the matching MusicXML file.
    2. Converts the MusicXML to an SMT string.
    3. Converts the PDF to a PNG image.
    4. Returns a dictionary for the dataset JSON.
    """
    print(f"Processing {pdf_path.name}...")
    xml_path = XML_INPUT_DIR / pdf_path.with_suffix('.musicxml').name
    png_path = IMAGE_OUTPUT_DIR / pdf_path.with_suffix('.png').name
    smt_path = SMT_OUTPUT_DIR / pdf_path.with_suffix('.smt').name

    if not xml_path.exists():
        print(f"  -> Skipping: No corresponding MusicXML file found at {xml_path}")
        return None

    # 1. Generate SMT from the ground-truth MusicXML
    smt_string = musicxml_to_smt(xml_path)
    if not smt_string:
        print(f"  -> Skipping: Failed to generate SMT from {xml_path.name}")
        return None

    # --- New: Save the generated SMT to a file for inspection ---
    try:
        SMT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(smt_path, 'w') as f:
            f.write(smt_string)
        print(f"  -> Ground-truth SMT saved to {smt_path.name}")
    except IOError as e:
        print(f"  -> Error writing SMT file: {e}")
        return None
    # --- End new section ---

    # 2. Convert the PDF to a PNG image
    try:
        IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'pdftoppm',
            '-png',
            '-singlefile',
            '-r', str(IMAGE_DPI),
            str(pdf_path),
            str(png_path.with_suffix('')) # pdftoppm adds the .png suffix
        ], check=True, capture_output=True, text=True)

        if not png_path.exists():
             print(f"  -> Error: PNG not created for {pdf_path.name}")
             return None
        
        print(f"  -> Successfully created image and SMT.")
        # Return paths relative to the training_data directory
        relative_image_path = png_path.relative_to(TRAINING_DATA_DIR)
        return {"image_path": str(relative_image_path), "st": smt_string}

    except subprocess.CalledProcessError as e:
        print(f"  -> Error during image rendering of {pdf_path.name}:")
        print(e.stderr)
        return None

def main():
    """
    Main function to generate the fine-tuning dataset.
    """
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset for RhythmForm.")
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for parallel processing (default: use all available cores)."
    )
    args = parser.parse_args()
    num_cores = args.cores or os.cpu_count() or 1

    # Find all PDF files in the input directory
    pdf_files = list(PDF_INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_INPUT_DIR}. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process with {num_cores} cores.")

    dataset = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        
        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Processing files"):
            result = future.result()
            if result:
                dataset.append(result)

    if not dataset:
        print("No data was successfully processed. Exiting.")
        return

    # Sort dataset by image path for consistency
    dataset.sort(key=lambda x: x['image_path'])

    # Save the final dataset manifest
    with open(DATASET_JSON_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nFine-tuning dataset creation complete. {len(dataset)} pairs created.")
    print(f"Manifest saved to {DATASET_JSON_PATH}")


if __name__ == '__main__':
    main()
