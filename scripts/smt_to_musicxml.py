import argparse
import re
from music21 import stream, note, chord, meter, duration, layout, clef, repeat, percussion

# --- SMT to music21 Mappings ---
# Maps SMT instrument abbreviations to MIDI pitches and display properties for a standard drum map.
INSTRUMENT_MAP = {
    "BD":  {'midi': 36, 'display_step': "F", 'display_octave': 4, 'notehead': 'normal'},
    "SD":  {'midi': 38, 'display_step': "C", 'display_octave': 5, 'notehead': 'normal'},
    "HH":  {'midi': 42, 'display_step': "G", 'display_octave': 5, 'notehead': 'x'},
    "HHO": {'midi': 46, 'display_step': "G", 'display_octave': 5, 'notehead': 'circle-x'},
    "CY":  {'midi': 49, 'display_step': "A", 'display_octave': 5, 'notehead': 'x'},
    "RD":  {'midi': 51, 'display_step': "B", 'display_octave': 5, 'notehead': 'x'},
    "LT":  {'midi': 45, 'display_step': "A", 'display_octave': 4, 'notehead': 'normal'},
    "MT":  {'midi': 47, 'display_step': "D", 'display_octave': 5, 'notehead': 'normal'},
    "HT":  {'midi': 50, 'display_step': "E", 'display_octave': 5, 'notehead': 'normal'},
    "FT":  {'midi': 41, 'display_step': "E", 'display_octave': 4, 'notehead': 'normal'},
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
    
    # Handle different types of barline tokens
    if token in ["barline", "measure_break"]:
        return {"type": "barline"}

    match = re.match(r"(\w+)\[(.*?)\]", token)
    if not match:
        return None

    token_type, value = match.groups()
    
    if token_type == "timeSignature":
        return {"type": "timeSignature", "value": value}
    elif token_type == "repeat" and value == "measure":
        return {"type": "repeat"}
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
    measure_number = 1
    current_measure = stream.Measure(number=measure_number)

    # Set up a default time signature and clef
    current_measure.append(meter.TimeSignature('4/4'))
    current_measure.append(clef.PercussionClef())

    print("Parsing tokens and building score...")
    for token_str in tokens:
        token = parse_token(token_str)
        if not token:
            continue

        if token["type"] == "barline":
            # Only append the measure if it has notes/rests in it
            if len(current_measure.notesAndRests) > 0:
                part.append(current_measure)
                if current_measure.number % 4 == 0:
                    part.append(layout.SystemLayout(isNew=True))
                
                measure_number += 1
                current_measure = stream.Measure(number=measure_number)
        
        elif token["type"] == "timeSignature":
            current_measure.append(meter.TimeSignature(token["value"]))

        elif token["type"] == "repeat":
            # If the current measure has content, append it first.
            if len(current_measure.notesAndRests) > 0:
                part.append(current_measure)
                if current_measure.number % 4 == 0:
                    part.append(layout.SystemLayout(isNew=True))
                measure_number += 1

            # Create a new measure specifically for the repeat mark
            repeat_measure = stream.Measure(number=measure_number)
            repeat_measure.append(repeat.RepeatMark())
            part.append(repeat_measure)
            if repeat_measure.number % 4 == 0:
                part.append(layout.SystemLayout(isNew=True))
            
            measure_number += 1
            # Start a new measure for subsequent notes
            current_measure = stream.Measure(number=measure_number)

        elif token["type"] == "rest":
            d = duration.Duration(DURATION_MAP.get(token["duration"], "quarter"))
            r = note.Rest(duration=d)
            current_measure.append(r)

        elif token["type"] == "note":
            d = duration.Duration(DURATION_MAP.get(token["duration"], "quarter"))
            
            note_objects = []
            # FIX: Remove duplicate instruments before creating notes
            unique_instruments = list(set(token["instruments"]))
            
            for inst_abbr in unique_instruments:
                inst_info = INSTRUMENT_MAP.get(inst_abbr)
                if inst_info:
                    # Use Unpitched for percussion notation
                    n = note.Unpitched()
                    n.duration = d
                    n.display_step = inst_info['display_step']
                    n.display_octave = inst_info['display_octave']
                    n.notehead = inst_info['notehead']
                    
                    # Add a stem direction hint for clarity
                    if n.display_step in ['F', 'G', 'A', 'B'] and n.display_octave == 4:
                        n.stemDirection = 'down'
                    else:
                        n.stemDirection = 'up'

                    note_objects.append(n)
            
            if len(note_objects) > 1:
                # Use a PercussionChord for multiple instruments at once
                c = percussion.PercussionChord(note_objects)
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
