import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pdf2image import convert_from_path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.omr_model.utils import musicxml_to_smt

def process_score(xml_path, pdf_path, output_dir, measures_per_page, write_smt=False):
    """
    Processes a single score:
    1. Converts PDF to images (one per page).
    2. Optionally converts XML to SMT.
    3. Splits SMT based on measures_per_page.
    4. Saves aligned image/(optional) SMT pairs.
    """
    try:
        base_name = xml_path.stem
        
        # 1. Convert PDF to Images
        if not pdf_path.exists():
            return []
            
        # Convert PDF to list of PIL images
        images = convert_from_path(str(pdf_path), dpi=200)
        
        full_smt = None
        if write_smt:
            # 2. Convert XML to SMT
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            
            full_smt = musicxml_to_smt(xml_content)
            if not full_smt:
                return []
                
            # Split Header and Body
            if '|' in full_smt:
                header, body = full_smt.split('|', 1)
                header = header.strip()
                body = body.strip()
            else:
                header = ""
                body = full_smt.strip()
                
            measures = [m.strip() for m in body.split('|')]
        
        # 3. Align Pages
        # We assume the PDF generation respected the page breaks inserted by generate_synthetic_scores.py
        # If measures_per_page is not set, we assume 1 page.
        
        dataset_entries = []
        
        # Determine chunk size
        # If SMT is not being written, we rely on the number of pages in the PDF
        if write_smt:
            chunk_size = measures_per_page if measures_per_page else len(measures)
            measure_chunks = [measures[i:i + chunk_size] for i in range(0, len(measures), chunk_size)]
            if len(images) != len(measure_chunks):
                # Mismatch between PDF pages and SMT chunks. 
                return []
        else:
            # If not writing SMT, we just process each image page.
            measure_chunks = [None] * len(images)

        for i, (image, chunk) in enumerate(zip(images, measure_chunks)):
            page_num = i + 1
            
            # Construct filenames
            image_filename = f"{base_name}_p{page_num}.png"
            image_path = output_dir / "images" / image_filename
            
            # Save Image
            image.save(image_path, "PNG")
            
            entry = {
                "image_path": str(image_path.relative_to(output_dir.parent))
            }
            
            if write_smt and chunk:
                smt_filename = f"{base_name}_p{page_num}.smt"
                smt_path = output_dir / "smt" / smt_filename
                
                if i == 0:
                    page_smt = f"{header} | {' | '.join(chunk)}"
                else:
                    minimal_header = "clef[percussion] time[4/4]"
                    page_smt = f"{minimal_header} | {' | '.join(chunk)}"
                
                with open(smt_path, 'w') as f:
                    f.write(page_smt)
                    
                entry["smt_path"] = str(smt_path.relative_to(output_dir.parent))
                entry["smt_string"] = page_smt
            
            dataset_entries.append(entry)
            
        return dataset_entries

    except Exception as e:
        print(f"Error processing {xml_path.name}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic dataset with multi-page support.")
    parser.add_argument('--data-dir', type=Path, required=True, help="Path to training_data directory")
    parser.add_argument('--cores', type=int, default=1, help="Number of cores")
    parser.add_argument('--measures-per-page', type=int, default=12, help="Measures per page used in generation")
    parser.add_argument('--write-smt', action='store_true', help="Write SMT files to training_data/smt directory")
    args = parser.parse_args()
    
    xml_dir = args.data_dir / "musicxml"
    pdf_dir = args.data_dir / "pdfs"
    
    # Create output dirs
    (args.data_dir / "images").mkdir(exist_ok=True)
    if args.write_smt:
        (args.data_dir / "smt").mkdir(exist_ok=True)
    
    xml_files = sorted(list(xml_dir.glob("*.xml")))
    tasks = []
    
    print(f"Processing {len(xml_files)} scores using {args.cores} cores...")
    
    dataset = []
    
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        for xml_file in xml_files:
            # Find corresponding PDF
            pdf_file = pdf_dir / xml_file.with_suffix('.pdf').name
            
            tasks.append(executor.submit(
                process_score, 
                xml_file, 
                pdf_file, 
                args.data_dir, 
                args.measures_per_page,
                args.write_smt
            ))
            
        for future in tqdm(as_completed(tasks), total=len(tasks)):
            result = future.result()
            dataset.extend(result)
            
    # Save dataset.json
    json_path = args.data_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset prepared. Saved {len(dataset)} samples to {json_path}")

if __name__ == '__main__':
    main()