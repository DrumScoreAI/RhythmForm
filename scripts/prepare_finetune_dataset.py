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

def process_musicxml_entry(xml_path, use_repeats):
    """
    Processes a single MusicXML file for the fine-tuning dataset:
    1. Checks for/generates the corresponding PDF file.
    2. Converts the MusicXML to an SMT string.
    3. Converts each page of the PDF to a PNG image.
    4. Returns a list of dictionaries for the dataset JSON, one for each page.
    """
    print(f"Processing {xml_path.name} (use_repeats={use_repeats})...")

    # --- 1. Ensure PDF exists, generate if missing ---
    pdf_path = PDF_INPUT_DIR / xml_path.with_suffix('.pdf').name
    if not pdf_path.exists():
        print(f"  -> PDF not found. Generating from {xml_path.name}...")
        try:
            PDF_INPUT_DIR.mkdir(parents=True, exist_ok=True)
            # Command: mscore -e -o <output.pdf> <input.musicxml>
            subprocess.run([
                'mscore',
                '-e',
                '-o', str(pdf_path),
                str(xml_path)
            ], check=True, capture_output=True, text=True)
            print(f"  -> Successfully generated {pdf_path.name}")
        except FileNotFoundError:
            print(f"  -> FATAL: 'mscore' command not found. Please ensure MuseScore is installed and in your PATH.")
            return []
        except subprocess.CalledProcessError as e:
            print(f"  -> Error during PDF generation for {xml_path.name}:")
            print(e.stderr)
            return []
    else:
        print(f"  -> Found existing PDF: {pdf_path.name}")

    # --- 2. Generate SMT from the ground-truth MusicXML ---
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            musicxml_content = f.read()
        smt_string = musicxml_to_smt(musicxml_content, use_repeats=use_repeats)
        if not smt_string:
            print(f"  -> Skipping: Failed to generate SMT from {xml_path.name}")
            return []
    except Exception as e:
        print(f"  -> Error reading or converting MusicXML {xml_path.name}: {e}")
        return []

    # --- 3. Convert the PDF to PNG images (one per page) ---
    try:
        IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        image_prefix = IMAGE_OUTPUT_DIR / xml_path.stem
        
        subprocess.run([
            'pdftoppm',
            '-png',
            '-r', str(IMAGE_DPI),
            str(pdf_path),
            str(image_prefix)
        ], check=True, capture_output=True, text=True)

        generated_images = sorted(list(IMAGE_OUTPUT_DIR.glob(f"{xml_path.stem}-*.png")))

        if not generated_images:
             print(f"  -> Error: No PNGs were created for {pdf_path.name}")
             return []
        
        print(f"  -> Successfully created {len(generated_images)} image(s) and SMT.")
        
        page_entries = []
        for png_path in generated_images:
            relative_image_path = png_path.relative_to(TRAINING_DATA_DIR)
            page_entries.append({"image_path": str(relative_image_path), "smt_string": smt_string})
        
        return page_entries

    except FileNotFoundError:
        print(f"  -> FATAL: 'pdftoppm' command not found. It is part of the poppler-utils package.")
        return []
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

    # Find all MusicXML files in the input directory
    xml_files = list(XML_INPUT_DIR.glob("*.musicxml")) + list(XML_INPUT_DIR.glob("*.xml"))
    if not xml_files:
        print(f"No MusicXML files found in {XML_INPUT_DIR}. Exiting.")
        return

    print(f"Found {len(xml_files)} MusicXML files to process with {num_cores} cores.")

    # Randomly decide for the whole batch if repeats should be used or not
    use_repeats_for_batch = random.choice([True, False])
    print(f"This batch will be processed with use_repeats={use_repeats_for_batch}.")

    dataset = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_xml = {executor.submit(process_musicxml_entry, xml, use_repeats_for_batch): xml for xml in xml_files}
        
        for future in tqdm(as_completed(future_to_xml), total=len(xml_files), desc="Processing files"):
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
