import argparse
import re
from music21 import stream, note, chord, meter, duration, layout, clef, repeat, metadata, instrument

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
    if token in ["barline", "measure_break", "|"]:
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

    score = stream.Score()
    md = metadata.Metadata()
    score.insert(0, md)

    part = stream.Part()
    part.insert(0, instrument.Percussion())
    part.insert(0, clef.PercussionClef())
    
    note_map = {
        'BD':  {'display_step': 'F', 'display_octave': 4, 'notehead': 'normal'},
        'SD':  {'display_step': 'C', 'display_octave': 5, 'notehead': 'normal'},
        'HH':  {'display_step': 'G', 'display_octave': 5, 'notehead': 'x'},
        'HHO': {'display_step': 'G', 'display_octave': 5, 'notehead': 'circle-x'},
        'CY':  {'display_step': 'A', 'display_octave': 5, 'notehead': 'x'},
        'RD':  {'display_step': 'B', 'display_octave': 5, 'notehead': 'x'},
        'LT':  {'display_step': 'A', 'display_octave': 4, 'notehead': 'normal'},
        'MT':  {'display_step': 'D', 'display_octave': 5, 'notehead': 'normal'},
        'HT':  {'display_step': 'E', 'display_octave': 5, 'notehead': 'normal'},
        'FT':  {'display_step': 'E', 'display_octave': 4, 'notehead': 'normal'},
    }

    # Split by spaces and handle the pipe character as a separate token.
    smt_content_with_spaces = smt_content.replace('|', ' | ')
    tokens = smt_content_with_spaces.replace('\n', ' ').split()

    time_signature = meter.TimeSignature('4/4')
    for token_str in tokens:
        token = parse_token(token_str)
        if token and token["type"] == "timeSignature":
            time_signature = meter.TimeSignature(token["value"])
            break
    part.append(time_signature)

    for token_str in tokens:
        token = parse_token(token_str)
        if not token: continue
        if token["type"] == "title": md.title = token["value"]
        elif token["type"] == "creator": md.composer = token["value"]

    current_measure_elements = []
    measure_number = 1
    current_offset = 0.0
    
    for token_str in tokens:
        token = parse_token(token_str)
        if not token or token["type"] in ["title", "creator", "timeSignature"]:
            continue

        if token["type"] == "barline":
            if current_measure_elements:
                m = stream.Measure(number=measure_number)
                for el in current_measure_elements:
                    m.insert(el['offset'], el['element'])
                
                # Automatically create voices for overlapping notes
                m.makeVoices(inPlace=True)

                part.append(m)
                
                measure_number += 1
                current_measure_elements = []
                current_offset = 0.0
                
                if (measure_number -1) % 4 == 0:
                    part.append(layout.SystemLayout(isNew=True))

        elif token["type"] == "rest":
            d = duration.Duration(type=DURATION_MAP.get(token["duration"], 'quarter'))
            r = note.Rest(duration=d)
            current_measure_elements.append({'offset': current_offset, 'element': r})
            current_offset += d.quarterLength

        elif token["type"] == "note":
            d = duration.Duration(type=DURATION_MAP.get(token["duration"], 'quarter'))
            
            elements_to_add = []
            for inst_abbr in token["instruments"]:
                if inst_abbr in INSTRUMENT_MAP:
                    inst_info = INSTRUMENT_MAP[inst_abbr]
                    
                    # Create an Unpitched note for percussion
                    n = note.Note()
                    n.duration = d
                    n.pitch.displayStep = inst_info['display_step']
                    n.pitch.displayOctave = inst_info['display_octave']
                    n.notehead = inst_info['notehead']
                    
                    # Assign the instrument to the note
                    p_inst = instrument.Percussion()
                    p_inst.instrumentName = inst_abbr
                    n.instruments = [p_inst]
                    
                    elements_to_add.append(n)

            if not elements_to_add:
                continue

            if len(elements_to_add) > 1:
                # For chords, music21 handles voices best when they are added as separate notes at the same offset
                for n in elements_to_add:
                    current_measure_elements.append({'offset': current_offset, 'element': n})
            else:
                current_measure_elements.append({'offset': current_offset, 'element': elements_to_add[0]})

            current_offset += d.quarterLength

    if current_measure_elements:
        m = stream.Measure(number=measure_number)
        for el in current_measure_elements:
            m.insert(el['offset'], el['element'])
        m.makeVoices(inPlace=True)
        part.append(m)

    score.insert(0, part)
    
    print(f"Writing MusicXML file to: {args.output_xml}")
    try:
        # The makeNotation call is critical for handling percussion clefs and voices correctly.
        score.makeNotation(inPlace=True)
        score.write('musicxml', fp=args.output_xml)
        print("Conversion successful.")
    except Exception as e:
        print(f"Error writing MusicXML file: {e}")


if __name__ == "__main__":
    main()
