import argparse
from pathlib import Path
import sys
import difflib

# --- Path Setup ---
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

from scripts.omr_model.utils import musicxml_to_smt

def main():
    parser = argparse.ArgumentParser(description="Perform a round-trip test for SMT -> MusicXML -> SMT conversion.")
    parser.add_argument('--name', type=str, required=True, help="The base name of the file to test (e.g., 'reach').")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    smt_dir = project_root / "training_data" / "fine_tuning" / "smt"
    xml_dir = project_root / "training_data" / "fine_tuning" / "musicxml"

    original_smt_path = smt_dir / f"{args.name}.smt"
    converted_xml_path = xml_dir / f"{args.name}.musicxml"

    if not original_smt_path.exists():
        print(f"Error: Original SMT file not found at {original_smt_path}")
        return
    if not converted_xml_path.exists():
        print(f"Error: Converted MusicXML file not found at {converted_xml_path}")
        return

    print(f"--- Original SMT from {original_smt_path.name} ---")
    with open(original_smt_path, 'r') as f:
        original_smt = f.read().strip()
    print(original_smt)

    print(f"\n--- Round-tripped SMT from {converted_xml_path.name} ---")
    try:
        with open(converted_xml_path, 'r', encoding='utf-8') as f:
            musicxml_content = f.read()
    except FileNotFoundError:
        print(f"Error: MusicXML file not found at {converted_xml_path}")
        return
    except Exception as e:
        print(f"Error reading file {converted_xml_path}: {e}")
        return
        
    round_trip_smt = musicxml_to_smt(musicxml_content).strip()
    print(round_trip_smt)

    print("\n--- Comparison ---")
    if original_smt == round_trip_smt:
        print("✅ Success: The original and round-tripped SMT are identical.")
    else:
        print("⚠️ Warning: The SMT files differ.")
        
        # Provide a diff to show the differences
        diff = difflib.unified_diff(
            original_smt.splitlines(keepends=True),
            round_trip_smt.splitlines(keepends=True),
            fromfile='original_smt',
            tofile='round_trip_smt',
        )
        print("\n--- Diff ---")
        for line in diff:
            print(line, end="")

if __name__ == "__main__":
    main()
