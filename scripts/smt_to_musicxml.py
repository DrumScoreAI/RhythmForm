import argparse
from pathlib import Path
import sys

# --- Path Setup ---
# This allows the script to be run from the root of the project, and also as a module.
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

from scripts.omr_model.utils import smt_to_musicxml

class SmtConverter:
    """A class to handle the conversion of an SMT string to a MusicXML file."""
    def __init__(self, smt_string):
        self.smt_string = smt_string
        self._score_obj = None

    def parse(self):
        """Uses the robust music21-based converter to get a Score object."""
        if self._score_obj is None:
            self._score_obj = smt_to_musicxml(self.smt_string)
        return self._score_obj
    
    def write_musicxml(self, output_path):
        """Parses the SMT to a Score object and writes it to a MusicXML file."""
        score_obj = self.parse()
        if score_obj:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # music21's write method handles file writing
                score_obj.write('musicxml', fp=str(output_path))
                return True
            except Exception as e:
                print(f"Error writing MusicXML file with music21: {e}")
                return False
        return False

def convert_smt_to_xml(smt_path, output_dir):
    """Converts a single SMT file to a MusicXML file using the SmtConverter."""
    try:
        with open(smt_path, 'r') as f:
            smt_string = f.read()

        converter = SmtConverter(smt_string)
        output_path = output_dir / smt_path.with_suffix('.xml').name
        if converter.write_musicxml(output_path):
            print(f"  -> Successfully converted {smt_path.name} to {output_path.name}")
        else:
            print(f"  -> Failed to convert {smt_path.name}")

    except Exception as e:
        print(f"  -> Error processing {smt_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert SMT files back to MusicXML.")
    parser.add_argument('--input-smt', type=str, required=True, help="Path to the input .smt file.")
    parser.add_argument('--output-musicxml', type=str, required=True, help="Path to save the output .musicxml file.")
    args = parser.parse_args()

    print(f"Reading SMT file from: {args.input_smt}")
    try:
        with open(args.input_smt, 'r') as f:
            smt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_smt}")
        return

    # Use the SmtConverter class
    converter = SmtConverter(smt_content)
    
    # Save the generated score object to a file
    output_path = Path(args.output_musicxml)
    if output_path.suffix.lower() != '.musicxml':
        print(f"Warning: Output file extension is not .musicxml. Changing to .musicxml")
        output_path = output_path.with_suffix('.musicxml')
    
    try:
        if converter.write_musicxml(output_path):
            print(f"Successfully converted SMT to MusicXML at: {output_path}")
        else:
            print(f"Failed to write MusicXML file for: {args.input_smt}")
    except Exception as e:
        print(f"Error writing MusicXML file: {e}")

if __name__ == '__main__':
    main()
