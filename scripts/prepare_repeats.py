import argparse
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm

def create_repeat_modified_xml(original_xml_path):
    """
    Checks for a companion .json file and, if it specifies repeats,
    injects <measure-style> tags for repeats into the XML file,
    saving the result to a new file with an '_altered' suffix.
    """
    json_path = original_xml_path.with_suffix('.json')
    if not json_path.exists():
        # No JSON file, so nothing to do.
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        repeated_measures = data.get("repeated_measures")
        if not repeated_measures:
            return None
    except (json.JSONDecodeError, IOError) as e:
        print(f"  -> Warning: Could not read or parse JSON {json_path.name}: {e}", file=sys.stderr)
        return None

    try:
        # Register namespace to avoid ET adding 'ns0:' prefixes
        ET.register_namespace('', "http://www.musicxml.org/xsd/musicxml.xsd")
        tree = ET.parse(original_xml_path)
        root = tree.getroot()
        
        part = root.find('.//part')
        if part is None:
            print(f"  -> [ERROR] Could not find a <part> element in {original_xml_path.name}", file=sys.stderr)
            return None

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
                    attributes.tail = "\n        "
                
                ms = ET.SubElement(attributes, 'measure-style')
                mr = ET.SubElement(ms, 'measure-repeat')
                mr.set('type', 'start')
                mr.text = '1'

            # The 'stop' tag goes on the measure AFTER the repeat block ends.
            stop_measure_num = group[-1] + 1
            stop_measure_element = part.find(f".//measure[@number='{stop_measure_num}']")

            if stop_measure_element is not None:
                attributes = stop_measure_element.find('attributes')
                if attributes is None:
                    attributes = ET.Element('attributes')
                    stop_measure_element.insert(0, attributes)
                    attributes.tail = "\n        "

                ms = ET.SubElement(attributes, 'measure-style')
                mr = ET.SubElement(ms, 'measure-repeat')
                mr.set('type', 'stop')

        # Save to a new file with _altered suffix
        altered_xml_path = original_xml_path.with_name(original_xml_path.stem + '_altered.xml')
        tree.write(altered_xml_path, encoding='UTF-8', xml_declaration=True)
        return str(altered_xml_path)

    except Exception as e:
        print(f"  -> XML modification failed for {original_xml_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function to find all XML files with companion JSON files and generate
    '_altered.xml' versions with repeat markings.
    """
    parser = argparse.ArgumentParser(description="Prepare XML files with repeat markings for rendering.")
    parser.add_argument(
        '--data-dir', 
        type=Path, 
        required=True, 
        help="Path to the training_data directory containing the 'musicxml' folder."
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=os.cpu_count(),
        help="Number of CPU cores to use."
    )
    args = parser.parse_args()

    xml_dir = args.data_dir / "musicxml"
    if not xml_dir.exists():
        print(f"Error: Directory not found: {xml_dir}", file=sys.stderr)
        sys.exit(1)

    xml_files = list(xml_dir.glob("*.xml"))
    # Exclude already altered files from being processed as originals
    xml_files = [f for f in xml_files if not f.stem.endswith('_altered')]
    
    print(f"Found {len(xml_files)} original XML files to check for repeats in {xml_dir}.")

    if not xml_files:
        print("No XML files to process.")
        return

    altered_files_count = 0
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        future_to_file = {executor.submit(create_repeat_modified_xml, xml_file): xml_file for xml_file in xml_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(xml_files), desc="Preparing repeat XMLs"):
            result = future.result()
            if result:
                altered_files_count += 1
    
    print(f"Finished. Created {altered_files_count} altered XML files for rendering.")


if __name__ == '__main__':
    # Set start method for multiprocessing to avoid issues on some platforms
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
