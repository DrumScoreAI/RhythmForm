import argparse
import re
from music21 import stream, note, chord, meter, duration, layout

# --- SMT to music21 Mappings ---
# Maps SMT instrument abbreviations to MIDI pitches for a standard drum map.
INSTRUMENT_MAP = {
    "BD": 36,  # Bass Drum
    "SD": 38,  # Snare Drum
    "HH": 42,  # Hi-Hat Closed
    "HHO": 46, # Hi-Hat Open
    "CY": 49,  # Crash Cymbal
    "RD": 51,  # Ride Cymbal
    "LT": 45,  # Low Tom
    "MT": 47,  # Mid Tom
    "HT": 50,  # High Tom
    "FT": 41,  # Floor Tom
}

# Maps SMT duration strings to music21 duration type strings.
DURATION_MAP = {
    "1": "whole",
    "1/2": "half",
    "1/4": "quarter",
    "1/8": "eighth",
    "1/16": "16th",
    "1/32": "32nd",
}

def parse_token(token):
    """Parses a single SMT token into a structured dictionary."""
    if not token:
        return None
    
    match = re.match(r"(\w+)\[(.*?)\]", token)
    if not match:
        if token == "barline":
            return {"type": "barline"}
        return None

    token_type, value = match.groups()
    
    if token_type == "timeSignature":
        return {"type": "timeSignature", "value": value}
    elif token_type == "rest":
        return {"type": "rest", "duration": value}
    elif token_type == "note":
        parts = value.split(',')
        instruments = parts[0].split('&')
        duration_str = parts[1]
        return {"type": "note", "instruments": instruments, "duration": duration_str}
    return None

def main():
    """Main function to convert an SMT file to MusicXML."""
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

    tokens = smt_content.replace('\n', ' ').split(' ')
    
    # --- Build music21 Score ---
    score = stream.Score()
    part = stream.Part()
    current_measure = stream.Measure()

    # Set up a default time signature
    current_measure.append(meter.TimeSignature('4/4'))

    print("Parsing tokens and building score...")
    for token_str in tokens:
        token = parse_token(token_str)
        if not token:
            continue

        if token["type"] == "barline":
            part.append(current_measure)
            current_measure = stream.Measure()
        
        elif token["type"] == "timeSignature":
            current_measure.append(meter.TimeSignature(token["value"]))

        elif token["type"] == "rest":
            d = duration.Duration(DURATION_MAP.get(token["duration"], "quarter"))
            r = note.Rest(duration=d)
            current_measure.append(r)

        elif token["type"] == "note":
            d = duration.Duration(DURATION_MAP.get(token["duration"], "quarter"))
            
            note_objects = []
            for inst_abbr in token["instruments"]:
                midi_pitch = INSTRUMENT_MAP.get(inst_abbr)
                if midi_pitch:
                    n = note.Note(midi_pitch, duration=d)
                    # You can add instrument information here if needed
                    # n.stemDirection = 'up' # or 'down'
                    note_objects.append(n)
            
            if len(note_objects) > 1:
                # Use a Chord for multiple instruments at once
                c = chord.Chord(note_objects)
                current_measure.append(c)
            elif len(note_objects) == 1:
                # Use a single Note
                current_measure.append(note_objects[0])

    # Append the last measure
    if current_measure.elements:
        part.append(current_measure)

    score.insert(0, part)
    
    print(f"Writing MusicXML file to: {args.output_xml}")
    score.write('musicxml', fp=args.output_xml)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
