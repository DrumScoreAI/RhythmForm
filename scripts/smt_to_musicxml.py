import argparse
from pathlib import Path
from scripts.omr_model.utils import smt_to_musicxml

def main():
    """Main function to convert an SMT file to MusicXML using the centralized utility."""
    parser = argparse.ArgumentParser(description="Convert a Symbolic Music Text (.smt) file to MusicXML.")
    parser.add_argument('--input-smt', type=str, required=True, help="Path to the input .smt file.")
    parser.add_argument('--output-xml', type=str, required=True, help="Path to save the output .musicxml file.")
    args = parser.parse_args()

    print(f"Reading SMT file from: {args.input_smt}")
    try:
        with open(args.input_smt, 'r') as f:
            smt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_smt}")
        return

    # Use the centralized conversion function
    score = smt_to_musicxml(smt_content)

    # Save the generated score object to a file
    try:
        output_path = Path(args.output_xml)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        score.write('musicxml', fp=output_path)
        print(f"Successfully converted SMT to MusicXML at: {args.output_xml}")
    except Exception as e:
        print(f"Error writing MusicXML file: {e}")

if __name__ == '__main__':
    main()
