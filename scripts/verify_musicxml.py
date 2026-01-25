import argparse
import sys
from pathlib import Path
import music21
from tqdm import tqdm
import os

# Suppress music21 configuration warnings
os.environ['MUSIC21_CONFIGURE_USER'] = '0'

def verify_file(xml_path):
    """
    Parses a MusicXML file and checks for basic validity.
    Returns (is_valid, message).
    """
    try:
        # 1. Attempt to parse
        score = music21.converter.parse(xml_path)
        
        # 2. Check for empty score
        if not score.parts:
            return False, "Score has no parts"
            
        # 3. Check measure consistency (basic check)
        # We iterate through measures to ensure they are accessible
        part = score.parts[0]
        measures = list(part.getElementsByClass('Measure'))
        
        if not measures:
            return False, "Part has no measures"
            
        # 4. Check for duration anomalies (optional, but good for OMR)
        # If the total duration is 0, something is likely wrong
        if score.duration.quarterLength == 0:
            return False, "Score has 0 duration"

        return True, "OK"

    except music21.Music21Exception as e:
        return False, f"Music21 Exception: {e}"
    except Exception as e:
        return False, f"General Exception: {e}"

def main():
    parser = argparse.ArgumentParser(description="Verify generated MusicXML files.")
    parser.add_argument(
        '--dir', 
        type=str, 
        required=True, 
        help="Directory containing .xml files"
    )
    args = parser.parse_args()
    
    xml_dir = Path(args.dir)
    if not xml_dir.exists():
        print(f"Directory not found: {xml_dir}")
        sys.exit(1)
        
    xml_files = list(xml_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {xml_dir}")
    
    valid_count = 0
    invalid_files = []
    
    for xml_file in tqdm(xml_files, desc="Verifying"):
        is_valid, msg = verify_file(xml_file)
        if is_valid:
            valid_count += 1
        else:
            invalid_files.append((xml_file.name, msg))
            
    print(f"\nVerification Complete.")
    print(f"Valid: {valid_count}/{len(xml_files)}")
    
    if invalid_files:
        print("\nInvalid Files:")
        for name, msg in invalid_files:
            print(f"  - {name}: {msg}")

if __name__ == '__main__':
    main()