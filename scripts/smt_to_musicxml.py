import argparse
import re
from music21 import stream, note, chord, meter, duration, layout, clef, repeat, percussion, metadata

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
    elif token_type in ["title", "creator"]:
        return {"type": token_type, "value": value}
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

    # Split the content by the page separator
    page_separator = "\n\n<page_break>\n\n"
    smt_pages = smt_content.split(page_separator)
    
    # --- Build music21 Score ---
    score = stream.Score()
    md = metadata.Metadata()
    score.insert(0, md)

    part = stream.Part()
    measure_number = 1
    
    # Set up a default time signature and clef for the first measure
    first_measure = stream.Measure(number=measure_number)
    first_measure.append(meter.TimeSignature('4/4'))
    first_measure.append(clef.PercussionClef())
    part.append(first_measure)
    
    current_measure = first_measure

    print(f"Found {len(smt_pages)} page(s) to process.")
    print("Parsing tokens and building score...")

    for page_idx, smt_page_content in enumerate(smt_pages):
        tokens = smt_page_content.replace('\n', ' ').split(' ')
        
        for token_str in tokens:
            token = parse_token(token_str)
            if not token:
                continue

            # Metadata is only processed from the first page
            if page_idx == 0:
                if token["type"] == "title":
                    md.title = token["value"]
                    continue
                elif token["type"] == "creator":
                    md.composer = token["value"]
                    continue
            
            # Skip metadata tokens on subsequent pages
            elif token["type"] in ["title", "creator"]:
                continue

            if token["type"] == "barline":
                # Only add a new measure if the current one is not empty
                if len(current_measure.notesAndRests) > 0:
                    part.append(current_measure)
                    measure_number += 1
                    current_measure = stream.Measure(number=measure_number)
            
            elif token["type"] == "timeSignature":
                # If the current measure is empty, add the time signature to it.
                # Otherwise, create a new measure for the time signature.
                if len(current_measure.notesAndRests) > 0:
                    part.append(current_measure)
                    measure_number += 1
                    current_measure = stream.Measure(number=measure_number)
                current_measure.append(meter.TimeSignature(token["value"]))

            elif token["type"] == "repeat":
                # Add a repeat barline to the end of the current measure
                current_measure.rightBarline = repeat.RepeatMark(direction='end')
                part.append(current_measure)
                
                # Start a new measure with a start repeat mark
                measure_number += 1
                current_measure = stream.Measure(number=measure_number)
                current_measure.leftBarline = repeat.RepeatMark(direction='start')

            elif token["type"] == "rest":
                duration_type = DURATION_MAP.get(token["duration"], "quarter")
                r = note.Rest(type=duration_type)
                current_measure.append(r)

            elif token["type"] == "note":
                duration_type = DURATION_MAP.get(token["duration"], "quarter")
                
                if len(token["instruments"]) > 1:
                    # Create a chord for multiple instruments
                    notes_for_chord = []
                    for inst_abbr in token["instruments"]:
                        inst_info = INSTRUMENT_MAP.get(inst_abbr)
                        if inst_info:
                            n = note.Note(inst_info['midi'])
                            n.notehead = inst_info['notehead']
                            notes_for_chord.append(n)
                    if notes_for_chord:
                        c = chord.Chord(notes_for_chord, type=duration_type)
                        current_measure.append(c)
                else:
                    # Create a single note
                    inst_abbr = token["instruments"][0]
                    inst_info = INSTRUMENT_MAP.get(inst_abbr)
                    if inst_info:
                        n = percussion.PercussionChord(type=duration_type)
                        n.notehead = inst_info['notehead']
                        
                        # Set display pitch
                        display_note = note.Note(f"{inst_info['display_step']}{inst_info['display_octave']}")
                        
                        # Create a new unpitched object for the instrument
                        unpitched = note.Unpitched(
                            displayName=inst_abbr
                        )
                        unpitched.displayStep = inst_info['display_step']
                        unpitched.displayOctave = inst_info['display_octave']
                        
                        # Assign the unpitched object and midi to the note
                        n.unpitched = unpitched
                        n.midi = inst_info['midi']

                        current_measure.append(n)

    # Append the last measure if it's not empty
    if len(current_measure.notesAndRests) > 0:
        part.append(current_measure)

    score.insert(0, part)
    
    # Add page and system breaks for better layout
    score.insert(0, layout.SystemLayout(isNew=True, top=150))

    print(f"Writing MusicXML file to: {args.output_xml}")
    try:
        score.write('musicxml', fp=args.output_xml)
        print("Conversion successful.")
    except Exception as e:
        print(f"Error writing MusicXML file: {e}")


if __name__ == "__main__":
    main()
