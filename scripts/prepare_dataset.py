import sys
from pathlib import Path
import os

# Add project root to sys.path to allow running from anywhere
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import csv
import music21
import subprocess
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from tqdm import tqdm
import argparse
import glob
from scripts.omr_model import config
from scripts.omr_model.utils import musicxml_to_smt

# --- Configuration ---
IMAGE_DPI = config.IMAGE_DPI
PROJECT_ROOT = config.PROJECT_ROOT
TRAINING_DATA_DIR = config.TRAINING_DATA_DIR
XML_DIR = config.XML_DIR
OUTPUT_IMAGE_DIR = config.DATA_IMAGES_DIR
PDF_OUTPUT_DIR = config.PDF_OUTPUT_DIR
SMT_OUTPUT_DIR = TRAINING_DATA_DIR / 'smt'
MANIFEST_FILE = config.MANIFEST_FILE
DATASET_JSON_PATH = config.DATASET_JSON_PATH
MUSESCORE_PATH = os.environ.get("MUSESCORE_PATH", "mscore4portable") # Use environment variable or default

def create_repeat_modified_xml(original_xml_path, repeated_measures):
    """
    Injects <measure-style> tags for repeats into an XML file and returns the path to a temporary file.
    """
    if not repeated_measures:
        return original_xml_path

    try:
        # Register namespace to avoid ET adding 'ns0:' prefixes
        ET.register_namespace('', "http://www.musicxml.org/xsd/musicxml.xsd")
        tree = ET.parse(original_xml_path)
        root = tree.getroot()
        
        part = root.find('.//part[@id="drumset"]')
        if part is None:
            part = root.find('.//part') # Fallback
        if part is None:
            print(f"  -> [ERROR] Could not find a <part> element in {original_xml_path.name}")
            return original_xml_path

        # Group consecutive repeated measures to handle start/stop tags correctly
        consecutive_repeats = []
        for m_num in sorted(repeated_measures):
            if not consecutive_repeats or m_num != consecutive_repeats[-1][-1] + 1:
                consecutive_repeats.append([m_num])
            else:
                consecutive_repeats[-1].append(m_num)

        for group in consecutive_repeats:
            start_measure_num = group[0]
            
            # Add 'start' tag to the first measure of the group
            start_measure_element = part.find(f".//measure[@number='{start_measure_num}']")
            if start_measure_element is not None:
                attributes = start_measure_element.find('attributes')
                if attributes is None:
                    attributes = ET.Element('attributes')
                    start_measure_element.insert(0, attributes)
                    # --- THIS IS THE FIX ---
                    # Add a newline and indentation in the 'tail' of the new attributes tag.
                    # This separates </attributes> from the following <note> tag.
                    attributes.tail = "\n        "
                
                ms = ET.SubElement(attributes, 'measure-style')
                mr = ET.SubElement(ms, 'measure-repeat')
                mr.set('type', 'start')
                mr.text = '1'

            # The 'stop' tag goes on the measure AFTER the repeat block ends.
            stop_measure_num = group[-1] + 1
            stop_measure_element = part.find(f".//measure[@number='{stop_measure_num}']")

            # Only add the stop tag if the measure actually exists.
            if stop_measure_element is not None:
                attributes = stop_measure_element.find('attributes')
                if attributes is None:
                    attributes = ET.Element('attributes')
                    stop_measure_element.insert(0, attributes)
                    # --- THIS IS THE FIX (applied here as well) ---
                    attributes.tail = "\n        "

                ms = ET.SubElement(attributes, 'measure-style')
                mr = ET.SubElement(ms, 'measure-repeat')
                mr.set('type', 'stop')

        # Save to a persistent debug file
        temp_xml_path = original_xml_path.with_name(original_xml_path.stem + '_altered.xml')
        tree.write(temp_xml_path, encoding='UTF-8', xml_declaration=True)
        print(f"  -> Saved altered XML for debugging: {temp_xml_path.name}")
        return temp_xml_path

    except Exception as e:
        print(f"  -> XML modification failed for {original_xml_path}: {e}")
        import traceback
        traceback.print_exc()
        return original_xml_path


def process_file(xml_path, write_smt=False):
    """
    Processes a single MusicXML file:
    1. Checks for a companion .json for repeat info.
    2. Generates ST, using 'repeat[bar,1]' token if needed (optionally writing to disk with `write_smt`).
    3. Creates a temporary, modified XML if repeats exist.
    4. Renders the appropriate XML to PDF and then to PNG.
    5. Cleans up temporary files.
    Returns a dictionary for the dataset.json or None on failure.
    """
    # The check for existing files is now handled efficiently in main() before this function is called.
    pdf_path = PDF_OUTPUT_DIR / xml_path.with_suffix('.pdf').name
    png_path = OUTPUT_IMAGE_DIR / xml_path.with_suffix('.png').name

    # --- 1. Handle repeats and ST generation ---
    json_path = xml_path.with_suffix('.json')
    repeated_measures = []
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                repeated_measures = data.get("repeated_measures", [])
        except Exception as e:
            # This is not a critical error, so we just print a warning.
            print(f"  -> Warning: Could not read or parse JSON {json_path.name}: {e}")

    # Extract title and creator from XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        title_element = root.find('.//movement-title')
        title = title_element.text.strip() if title_element is not None and title_element.text else "Music21 Fragment"

        creator_element = root.find('.//creator[@type="composer"]')
        if creator_element is None:
            creator_element = root.find('.//creator') # Fallback
        creator = creator_element.text.strip() if creator_element is not None and creator_element.text else "Music21"

    except ET.ParseError as e:
        print(f"  -> XML parsing error for {xml_path.name}: {e}")
        title = "Music21 Fragment"
        creator = "Music21"

    smt_string = musicxml_to_smt(xml_path)
    if not smt_string:
        return None

    # Prepend dynamic metadata to the sequence
    smt_string = f"title[{title}] creator[{creator}] " + smt_string
    
    if write_smt:
        smt_output_path = SMT_OUTPUT_DIR / xml_path.with_suffix('.smt').name
        try:
            with open(smt_output_path, 'w') as f:
                f.write(smt_string)
        except Exception as e:
            print(f"  -> Warning: Could not write SMT file {smt_output_path.name}: {e}")

    # --- 2. Render Image (with repeats if necessary) ---
    xml_to_render = xml_path
    if repeated_measures:
        # This function returns the path to the new _altered.xml file
        xml_to_render = create_repeat_modified_xml(xml_path, repeated_measures)

    try:
        # Render the (potentially temporary) XML to PDF
        subprocess.run([
            'xvfb-run', 
            '-a',
            MUSESCORE_PATH,
            '-o', str(pdf_path),
            str(xml_to_render)
        ], check=True, capture_output=True, text=True)

        # Convert PDF to PNG
        subprocess.run([
            'pdftoppm',
            '-png',
            '-singlefile',
            '-r', str(IMAGE_DPI), # DPI
            str(pdf_path),
            str(png_path.with_suffix('')) # pdftoppm adds the suffix
        ], check=True, capture_output=True, text=True)

        if not png_path.exists():
             print(f"  -> Error: PNG not created for {xml_path.name}")
             return None

        return {"image_path": str(png_path.relative_to(TRAINING_DATA_DIR)), "smt_string": smt_string}

    except subprocess.CalledProcessError as e:
        print(f"  -> Error during rendering of {xml_to_render.name}:")
        print(e.stderr)
        return None
    finally:
        pass
        # --- 3. Cleanup ---
        # Keep the altered XML for debugging, but remove the intermediate PDF.
        # if pdf_path.exists():
        #     pdf_path.unlink()


def process_chunk(xml_paths_chunk, write_smt=False):
    """
    Worker function to process a chunk of XML files.
    This runs in a separate process.
    """
    results = []
    for xml_path in xml_paths_chunk:
        result = process_file(xml_path, write_smt)
        if result:
            results.append(result)
    return results

def main():
    """
    Main function to generate the dataset using a manifest file.
    It reads training_data.csv, processes each entry in parallel to generate
    """

    # --- Add argparse for --cores argument ---
    parser = argparse.ArgumentParser(description="Prepare dataset for RhythmForm.")
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for parallel processing (default: use all available cores)."
    )
    parser.add_argument(
        "--write-smt",
        action="store_true",
        help="Write SMT files to training_data/smt directory."
    )
    args = parser.parse_args()
    num_cores = args.cores or os.cpu_count() or 1
    if args.write_smt:
        write_smt_files = args.write_smt
    else:
        write_smt_files = False

    # Ensure output directories exist
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if write_smt_files:
        SMT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_FILE.exists():
        print(f"Error: Manifest file not found at {MANIFEST_FILE}")
        return

    # Read the manifest to get the list of files to process
    xml_paths_to_process = []
    with open(MANIFEST_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # We are currently only processing "drums-only" files
            if row.get('do_or_mi') == 'do' and row.get('n_or_p') == 'n' and row.get('musicxml'):
                xml_path = XML_DIR / row['musicxml']
                if xml_path.exists():
                    xml_paths_to_process.append(xml_path)
                else:
                    print(f"  [WARNING] File '{row['musicxml']}' listed in manifest but not found in {XML_DIR}. Skipping.")
            elif row.get('n_or_p') == 'p':
                pass # We no longer need to print this for every processed file.
            elif row.get('do_or_mi') == 'do':
                print(f"  [WARNING] 'musicxml' field missing for drums-only entry in manifest. Skipping.")
            elif row.get('do_or_mi') == 'mi':
                print(f"  [INFO] Skipping 'mi' entry. Multi-instrument files are not yet supported.")

    print(f"Found {len(xml_paths_to_process)} 'drums-only' files to process from manifest.")

    # --- OPTIMIZATION ---
    # The original check was very slow because it performed a list lookup for every file.
    # By creating a set of existing PNG filenames first, we can do a much faster O(1) lookup.
    print("Checking for already processed files to skip...")
    existing_png_files = {Path(f).name for f in glob.glob(str(OUTPUT_IMAGE_DIR / '*.png'))}
    print(f"Found {len(existing_png_files)} existing PNG files in the image directory.")

    original_count = len(xml_paths_to_process)
    # Now, filter the list using the fast set lookup
    xml_paths_to_process = [
        p for p in xml_paths_to_process 
        if p.with_suffix('.png').name not in existing_png_files
    ]
    
    filtered_count = original_count - len(xml_paths_to_process)
    print(f"Skipping {filtered_count} files that have already been rendered to PNG.")
    print(f"Number of new files to process: {len(xml_paths_to_process)}")
    # --- END OPTIMIZATION ---

    if not xml_paths_to_process:
        print("No new files to process. Exiting.")
        return

    if os.path.exists(DATASET_JSON_PATH):
        with open(DATASET_JSON_PATH, 'r') as f:
            try:
                current_dataset = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {DATASET_JSON_PATH}. Starting with an empty dataset.")
                current_dataset = []
    else:
        current_dataset = []
    
    dataset = current_dataset[:] # Start with a copy of the existing dataset

    # --- CHUNKING OPTIMIZATION ---
    # Split the work into chunks for more efficient parallel processing.
    chunk_size = max(1, len(xml_paths_to_process) // num_cores)
    chunks = [xml_paths_to_process[i:i + chunk_size] for i in range(0, len(xml_paths_to_process), chunk_size)]
    print(f"Splitting {len(xml_paths_to_process)} files into {len(chunks)} chunks for {num_cores} cores.")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit each chunk to a worker
        future_to_chunk = {executor.submit(process_chunk, chunk, write_smt_files): chunk for chunk in chunks}
        
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Processing chunks"):
            chunk_results = future.result()
            if chunk_results:
                dataset.extend(chunk_results)
    # --- END CHUNKING OPTIMIZATION ---

    # Sort dataset by image path for consistency
    dataset.sort(key=lambda x: x['image_path'])

    # Save the final dataset manifest
    with open(DATASET_JSON_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset creation complete. {len(dataset)} pairs created.")
    print(f"Manifest saved to {DATASET_JSON_PATH}")


if __name__ == '__main__':
    main()
