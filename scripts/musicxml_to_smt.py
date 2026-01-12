import argparse
from pathlib import Path
import sys

# --- Path Setup ---
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

from scripts.omr_model.utils import musicxml_to_smt

def main():
    parser = argparse.ArgumentParser(description="Convert MusicXML files to SMT.")
    parser.add_argument('--input-musicxml', type=str, required=True, help="Path to the input .musicxml file.")
    parser.add_argument('--output-smt', type=str, required=True, help="Path to save the output .smt file.")
    args = parser.parse_args()

    try:
        with open(args.input_musicxml, 'r', encoding='utf-8') as f:
            musicxml_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_musicxml}")
        return
    except Exception as e:
        print(f"Error reading file {args.input_musicxml}: {e}")
        return

    smt_content = musicxml_to_smt(musicxml_content)

    output_path = Path(args.output_smt)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(smt_content)
        print(f"Successfully converted MusicXML to SMT at: {output_path}")
    except Exception as e:
        print(f"Error writing SMT file: {e}")

if __name__ == '__main__':
    main()
