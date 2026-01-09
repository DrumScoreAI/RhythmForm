import os
from pathlib import Path
import music21
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from tqdm import tqdm
import argparse
import random
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

def process_pdf(pdf_path, use_repeats):
    """
    Processes a single PDF file for the fine-tuning dataset:
    1. Finds the matching MusicXML file.
    2. Converts the MusicXML to an SMT string.
    3. Converts each page of the PDF to a PNG image.
    4. Returns a list of dictionaries for the dataset JSON, one for each page.
    """
    print(f"Processing {pdf_path.name} (use_repeats={use_repeats})...")
    xml_filename = pdf_path.with_suffix('.xml').name
    # The conversion script might create .xml, but let's also check for .musicxml
    xml_path = XML_INPUT_DIR / xml_filename
    if not xml_path.exists():
        xml_path = XML_INPUT_DIR / pdf_path.with_suffix('.musicxml').name

    if not xml_path.exists():
        print(f"  -> Skipping: No corresponding MusicXML file found for {pdf_path.name}")
        return []

    # 1. Generate SMT from the ground-truth MusicXML
    smt_string = musicxml_to_smt(xml_path, use_repeats=use_repeats)
    if not smt_string:
        print(f"  -> Skipping: Failed to generate SMT from {xml_path.name}")
        return []

    # --- Save the generated SMT to a file for inspection ---
    try:
        SMT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        smt_path = SMT_OUTPUT_DIR / pdf_path.with_suffix('.smt').name
        with open(smt_path, 'w') as f:
            f.write(smt_string)
        print(f"  -> Ground-truth SMT saved to {smt_path.name}")
    except IOError as e:
        print(f"  -> Error writing SMT file: {e}")
        return []
    # --- End new section ---

    # 2. Convert the PDF to PNG images (one per page)
    try:
        IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # The output path for pdftoppm should be a prefix. It will add '-1.png', '-2.png', etc.
        image_prefix = IMAGE_OUTPUT_DIR / pdf_path.stem
        
        subprocess.run([
            'pdftoppm',
            '-png',
            # Removed '-singlefile' to get all pages
            '-r', str(IMAGE_DPI),
            str(pdf_path),
            str(image_prefix)
        ], check=True, capture_output=True, text=True)

        # Find all generated images for this PDF
        generated_images = sorted(list(IMAGE_OUTPUT_DIR.glob(f"{pdf_path.stem}-*.png")))

        if not generated_images:
             print(f"  -> Error: No PNGs were created for {pdf_path.name}")
             return []
        
        print(f"  -> Successfully created {len(generated_images)} image(s) and SMT.")
        
        page_entries = []
        for png_path in generated_images:
            relative_image_path = png_path.relative_to(TRAINING_DATA_DIR)
            page_entries.append({"image_path": str(relative_image_path), "st": smt_string})
        
        return page_entries

    except subprocess.CalledProcessError as e:
        print(f"  -> Error during image rendering of {pdf_path.name}:")
        print(e.stderr)
        return []

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

    # Randomly decide for the whole batch if repeats should be used or not
    use_repeats_for_batch = random.choice([True, False])
    print(f"This batch will be processed with use_repeats={use_repeats_for_batch}.")

    dataset = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf, use_repeats_for_batch): pdf for pdf in pdf_files}
        
        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Processing files"):
            results = future.result()
            if results:
                dataset.extend(results)

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
