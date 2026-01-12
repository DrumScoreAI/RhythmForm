import music21
from fractions import Fraction
from bs4 import BeautifulSoup
import re

# This file contains shared utility functions for the OMR model,
# primarily for converting between MusicXML and Symbolic Music Text (SMT) formats.

# --- Drum MIDI to SMT and Display-Step to SMT Mappings ---
DRUM_MIDI_TO_SMT = {
    # Bass Drums
    35: 'BD', 36: 'BD',
    # Snares
    38: 'SD', 40: 'SD', 37: 'SD', # Map side stick to snare
    # Toms
    41: 'FT', 43: 'FT', 45: 'LT', 47: 'MT', 48: 'HT', 50: 'HT',
    # Hi-Hats
    42: 'HH', 44: 'HH', 46: 'HHO', # Closed, Pedal, Open
    # Cymbals
    49: 'CY', 57: 'CY', 51: 'RD', 59: 'RD', 55: 'CY', 52: 'CY', # Crash, Splash, Chinese
}

# This maps the visual representation (staff line, octave, and notehead) back to an SMT token.
# This is crucial for parsing generated scores that use <unpitched>.
# The key is a tuple: (displayStep, displayOctave, notehead)
DRUM_DISPLAY_TO_SMT = {
    # Bass Drum on F, octave 4
    ('F', 4, 'normal'): 'BD',
    # Snare on C, octave 5
    ('C', 5, 'normal'): 'SD',
    # Side Stick on C, octave 5 with x notehead
    ('C', 5, 'x'): 'SD',
    # Closed Hi-Hat on G, octave 5 with x notehead
    ('G', 5, 'x'): 'HH',
    # Open Hi-Hat on G, octave 5 with circle-x notehead
    ('G', 5, 'circle-x'): 'HHO',
    # Pedal Hi-Hat on E, octave 4 with x notehead
    ('E', 4, 'x'): 'HH',
    # Crash Cymbal on A, octave 5 with cross notehead
    ('A', 5, 'cross'): 'CY',
    # Ride Cymbal on B, octave 5 with cross notehead
    ('B', 5, 'cross'): 'RD',
    # Low Tom on A, octave 4
    ('A', 4, 'normal'): 'FT',
    # High Tom on E, octave 5
    ('E', 5, 'normal'): 'HT',
}

# --- SMT to MusicXML Mappings (Inverse of DRUM_DISPLAY_TO_SMT) ---
SMT_TO_DRUM_DISPLAY = {
    'BD':  {'display_step': 'F', 'display_octave': 4, 'notehead': 'normal'},
    'SD':  {'display_step': 'C', 'display_octave': 5, 'notehead': 'normal'},
    'HH':  {'display_step': 'G', 'display_octave': 5, 'notehead': 'x'},
    'HHO': {'display_step': 'G', 'display_octave': 5, 'notehead': 'circle-x'},
    'CY':  {'display_step': 'A', 'display_octave': 5, 'notehead': 'cross'},
    'RD':  {'display_step': 'B', 'display_octave': 5, 'notehead': 'cross'},
    'FT':  {'display_step': 'A', 'display_octave': 4, 'notehead': 'normal'}, # Floor Tom
    'HT':  {'display_step': 'E', 'display_octave': 5, 'notehead': 'normal'}, # High Tom
    # Add other mappings as needed to be a complete inverse
    'LT':  {'display_step': 'C', 'display_octave': 5, 'notehead': 'normal'}, # Low Tom (example, adjust)
    'MT':  {'display_step': 'B', 'display_octave': 4, 'notehead': 'normal'}, # Mid Tom (example, adjust)
}


def _get_duration_map_from_xml(soup: BeautifulSoup):
    """
    Parses a BeautifulSoup object of a MusicXML to create a mapping from measure 
    and element index to its correct fractional duration.
    The key is a tuple (measure_number, element_index_in_measure), and the
    value is the duration as a Fraction.
    """
    duration_map = {}
    try:
        # MusicXML can define divisions in <part-wise><part><measure><attributes><divisions>
        # We will assume the first one found is the one to use for the whole score.
        first_divisions_tag = soup.find('divisions')
        divisions = int(first_divisions_tag.string) if first_divisions_tag and first_divisions_tag.string else 1

        all_measures = soup.find_all('measure')
        for measure_elem in all_measures:
            measure_number = int(measure_elem.get('number', 0))
            
            # A measure can override the default divisions
            attributes = measure_elem.find('attributes')
            if attributes:
                measure_divisions_tag = attributes.find('divisions')
                if measure_divisions_tag and measure_divisions_tag.string:
                    divisions = int(measure_divisions_tag.string)
            
            element_idx = 0
            # Find all direct children 'note' of the measure
            for element in measure_elem.find_all(['note', 'forward', 'backup'], recursive=False):
                # Skip chord elements as they don't advance time in the same way
                if element.name == 'note' and element.find('chord'):
                    continue

                duration_tag = element.find('duration')
                if duration_tag and duration_tag.string:
                    raw_duration = int(duration_tag.string)
                    if divisions > 0:
                        duration = Fraction(raw_duration, divisions).limit_denominator()
                        duration_map[(measure_number, element_idx)] = duration
                
                element_idx += 1
                    
    except Exception as e:
        print(f"Warning: Could not build duration map from XML content: {e}")
        
    return duration_map


def _convert_bs_notes_to_smt(note_elements, duration):
    """
    Converts a list of BeautifulSoup <note> elements (representing a single note or a chord)
    to a single SMT token.
    """
    smt_notes = []
    for note_elem in note_elements:
        # Handle rests
        if note_elem.find('rest'):
            # This function is for notes, but we can safeguard.
            # In the main loop, rests are handled separately.
            # If a chord contains a rest, we'll represent the whole event as a rest.
            return f"rest[{duration}]"

        # Handle unpitched notes (percussion)
        unpitched = note_elem.find('unpitched')
        if unpitched:
            display_step_tag = unpitched.find('display-step')
            display_octave_tag = unpitched.find('display-octave')
            
            if display_step_tag and display_octave_tag:
                display_step = display_step_tag.string
                display_octave = int(display_octave_tag.string)
                notehead_tag = note_elem.find('notehead')
                notehead_text = notehead_tag.string if notehead_tag and notehead_tag.string else 'normal'
                
                display_key = (display_step, display_octave, notehead_text)
                if display_key in DRUM_DISPLAY_TO_SMT:
                    smt_notes.append(DRUM_DISPLAY_TO_SMT[display_key])
    
    if not smt_notes:
        return None

    note_str = ".".join(sorted(list(set(smt_notes))))
    return f"note[{note_str},{duration}]"


def musicxml_to_smt(musicxml_content: str, use_repeats=False) -> str:
    """
    Converts a MusicXML file content to its SMT representation using BeautifulSoup.
    This is the main entry point for MusicXML to SMT conversion.

    Args:
        musicxml_content (str): The string content of the MusicXML file.
        use_repeats (bool): If True, will try to find repeated measures and
                            represent them as 'repeat[measure]'. If False,
                            it will "unroll" any repeats into full note sequences.
    """
    try:
        soup = BeautifulSoup(musicxml_content, 'xml')
    except Exception as e:
        print(f"Error parsing XML content with BeautifulSoup: {e}")
        return ""

    # This is the most reliable way to get durations, so we do it first.
    duration_map = _get_duration_map_from_xml(soup)

    header_tokens = []
    # --- Extract Metadata ---
    # Try to get metadata from <credit> tags first, as they are often more detailed.
    # This is what is actually rendered on the score.
    title = None
    composer = None
    subtitle = None

    for credit_tag in soup.find_all('credit'):
        credit_type_tag = credit_tag.find('credit-type')
        credit_words_tag = credit_tag.find('credit-words')
        if credit_type_tag is not None and credit_words_tag is not None and credit_words_tag.string:
            text = credit_words_tag.string.strip()
            if credit_type_tag.string == 'title' and not title:
                title = text
            elif credit_type_tag.string == 'composer' and not composer:
                composer = text
            elif credit_type_tag.string == 'subtitle' and not subtitle:
                subtitle = text

    if title:
        header_tokens.append(f"title[{title}]")
    if subtitle:
        header_tokens.append(f"subtitle[{subtitle}]")
    if composer:
        header_tokens.append(f"composer[{composer}]")

    # Find the first clef and time signature in the first part.
    first_part = soup.find('part')
    if first_part:
        # Get Clef
        clef_tag = first_part.find('clef')
        if clef_tag and clef_tag.find('sign') and clef_tag.find('sign').string == 'percussion':
            header_tokens.append("clef[percussion]")

        # Get Time Signature
        time_tag = first_part.find('time')
        if time_tag and time_tag.find('beats') and time_tag.find('beat-type'):
            beats = time_tag.find('beats').string
            beat_type = time_tag.find('beat-type').string
            header_tokens.append(f"time[{beats}/{beat_type}]")

    smt_measures = []
    previous_measure_smt = None
    last_section_text = None

    all_measures = soup.find_all('measure')
    for measure_elem in all_measures:
        measure_number = int(measure_elem.get('number', 0))
        
        # Check for measure repeat symbol. MuseScore can export this in a few ways.
        # The most common for '%' is <measure-style><measure-repeat type="start" count="1"/></measure-style>
        # Or a simple <repeat type="percent"/>. We'll check for both.
        is_repeat_measure = False
        if measure_elem.find('measure-repeat') or (measure_elem.find('repeat') and measure_elem.find('repeat').get('type') == 'percent'):
             is_repeat_measure = True

        if use_repeats and is_repeat_measure:
            measure_smt = "repeat[measure]"
        elif not use_repeats and is_repeat_measure:
            measure_smt = previous_measure_smt
        else:
            # This is a normal measure, so we generate its SMT.
            measure_smt_tokens = []
            element_idx = 0
            
            # Find all direct children 'note', 'forward', 'backup', and 'direction' of the measure
            for element in measure_elem.find_all(['note', 'forward', 'backup', 'direction'], recursive=False):
                # Handle text annotations like "Verse"
                if element.name == 'direction':
                    words = element.find('words')
                    if words and words.string:
                        current_text = words.string.strip()
                        if current_text != last_section_text:
                            measure_smt_tokens.append(f"text[{current_text}]")
                            last_section_text = current_text
                    continue

                # Skip chord elements as they are handled with the main note
                if element.name == 'note' and element.find('chord'):
                    continue

                duration = duration_map.get((measure_number, element_idx), Fraction(0))
                
                # We need to handle chords, which are multiple notes at the same time position.
                # The first note does not have a <chord/> tag, subsequent ones do.
                if element.name == 'note':
                    # Check for tuplets
                    tuplet_start = element.find('tuplet', {'type': 'start'})
                    if tuplet_start and tuplet_start.get('number'):
                        measure_smt_tokens.append(f"tuplet[{tuplet_start.get('number')}:start]")

                    if element.find('rest'): # It's a rest within a note tag
                        measure_smt_tokens.append(f"rest[{duration}]")
                    else:
                        chord_notes = [element]
                        next_sibling = element.find_next_sibling('note')
                        while next_sibling and next_sibling.find('chord'):
                            chord_notes.append(next_sibling)
                            next_sibling = next_sibling.find_next_sibling('note')
                        
                        token = _convert_bs_notes_to_smt(chord_notes, duration)
                        if token:
                            measure_smt_tokens.append(token)
                    
                    tuplet_stop = element.find('tuplet', {'type': 'stop'})
                    if tuplet_stop and tuplet_stop.get('number'):
                        measure_smt_tokens.append(f"tuplet[{tuplet_stop.get('number')}:stop]")

                element_idx += 1
            
            measure_smt = " ".join(measure_smt_tokens)

        if measure_smt:
            smt_measures.append(measure_smt)
        
        if not is_repeat_measure:
            previous_measure_smt = measure_smt

    header_str = " ".join(header_tokens)
    body_str = " | ".join(smt_measures)

    if header_str and body_str:
        return f"{header_str} | {body_str}"
    elif body_str:
        return body_str
    else:
        return ""


def _parse_smt_token(token_str):
    """Parses a single SMT token string into a dictionary."""
    match = re.match(r"(\w+)\[([^\]]*)\]", token_str)
    if not match:
        return None

    token_type = match.group(1)
    content = match.group(2)
    
    token_info = {"type": token_type}

    if token_type in ["title", "subtitle", "composer", "creator", "clef", "time"]:
        token_info["value"] = content
    elif token_type == "text":
        token_info["value"] = content
    elif token_type == "rest":
        # Handle cases where duration might be missing or invalid
        duration_str = content.strip()
        if not duration_str:
            token_info["duration"] = "0.0" # Assign a default valid duration
        else:
            token_info["duration"] = duration_str
    elif token_type == "note":
        parts = content.split(',')
        instruments = parts[0].strip().split('.')
        token_info["instruments"] = [inst for inst in instruments if inst]
        
        duration_str = parts[1].strip() if len(parts) > 1 else ""
        if not duration_str:
            token_info["duration"] = "0.0" # Assign a default valid duration
        else:
            token_info["duration"] = duration_str

    elif token_type == "tuplet":
        parts = content.split(',')
        if len(parts) == 2:
            token_info["actual_notes"] = parts[0].strip()
            token_info["normal_notes"] = parts[1].strip()

    return token_info


def smt_to_musicxml(smt_string: str) -> music21.stream.Score:
    """
    Converts an SMT string to a music21 Score object.
    """
    try:
        score = music21.stream.Score()
        part = music21.stream.Part()
        
        # Split string into header and body
        parts = smt_string.split('|', 1)
        header_string = parts[0].strip()
        body_string = parts[1].strip() if len(parts) > 1 else ""

        header_tokens = header_string.split()
        
        # --- Initial setup from header ---
        for token_str in header_tokens:
            token = _parse_smt_token(token_str)
            if not token: continue
            
            if token["type"] == "title":
                score.metadata = music21.metadata.Metadata()
                score.metadata.title = token["value"]
            elif token["type"] == "subtitle":
                if not score.metadata:
                    score.metadata = music21.metadata.Metadata()
                score.metadata.movementName = token["value"]
            elif token["type"] == "creator":
                if not score.metadata:
                    score.metadata = music21.metadata.Metadata()
                score.metadata.composer = token["value"]
            elif token["type"] == "clef" and token["value"] == "percussion":
                part.insert(0, music21.clef.PercussionClef())
            elif token["type"] == "time":
                part.insert(0, music21.meter.TimeSignature(token["value"]))

        # --- Process measures from body ---
        measure_strings = body_string.split('|')
        measure_number = 1
        tuplet_state = None

        for measure_str in measure_strings:
            current_measure = music21.stream.Measure(number=measure_number)
            tokens = measure_str.strip().split()
            
            if not tokens:
                continue

            for token_str in tokens:
                token = _parse_smt_token(token_str)
                if not token:
                    continue
                
                if token["type"] == "text":
                    # Create a text expression and add it to the measure
                    te = music21.expressions.TextExpression(token["value"])
                    current_measure.append(te)
                    continue # Move to the next token

                if token["type"] == "tuplet":
                    # This logic needs to be improved to handle nested tuplets if they exist
                    if "start" in token.get("value", ""):
                        tuplet_state = music21.duration.Tuplet(numberNotes=3, notesFollowing=1) # Basic triplet
                    else: # stop
                        tuplet_state = None
                    continue

                elif token["type"] == "rest":
                    r = music21.note.Rest()
                    try:
                        r.duration.quarterLength = float(token["duration"])
                    except ValueError:
                        r.duration.quarterLength = Fraction(token["duration"])
                    if r.quarterLength == 0.0:
                        r.duration.type = 'zero'
                    if tuplet_state:
                        r.duration.appendTuplet(tuplet_state)
                    current_measure.append(r)

                elif token["type"] == "note":
                    duration = music21.duration.Duration()
                    try:
                        duration.quarterLength = float(token["duration"])
                    except ValueError:
                        duration.quarterLength = Fraction(token["duration"])
                    if duration.quarterLength == 0.0:
                        duration.type = 'zero'
                    
                    if tuplet_state:
                        duration.appendTuplet(tuplet_state)

                    if len(token["instruments"]) > 1:
                        # It's a chord
                        chord_notes = []
                        for instrument_name in token["instruments"]:
                            if instrument_name in SMT_TO_DRUM_DISPLAY:
                                display_info = SMT_TO_DRUM_DISPLAY[instrument_name]
                                n = music21.note.Unpitched()
                                n.display_step = display_info['display_step']
                                n.display_octave = display_info['display_octave']
                                if 'notehead' in display_info:
                                    n.notehead = display_info['notehead']
                            else:
                                n = music21.note.Note()
                                n.pitch.midi = int(instrument_name)
                            chord_notes.append(n)
                        
                        c = music21.chord.Chord(chord_notes)
                        c.duration = duration
                        current_measure.append(c)
                    elif len(token["instruments"]) == 1:
                        # It's a single note
                        instrument_name = token["instruments"][0]
                        if instrument_name in SMT_TO_DRUM_DISPLAY:
                            display_info = SMT_TO_DRUM_DISPLAY[instrument_name]
                            n = music21.note.Unpitched()
                            n.display_step = display_info['display_step']
                            n.display_octave = display_info['display_octave']
                            if 'notehead' in display_info:
                                n.notehead = display_info['notehead']
                        else:
                            try:
                                n = music21.note.Note()
                                n.pitch.midi = int(instrument_name)
                            except (ValueError, TypeError):
                                n = music21.note.Rest()

                        n.duration = duration
                        current_measure.append(n)

            if current_measure.hasElements():
                part.append(current_measure)
                measure_number += 1

        score.insert(0, part)
        return score
    except Exception as e:
        import traceback
        print(f"Error processing SMT string: {e}")
        print(traceback.format_exc())
        return None


def smt_to_musicxml_manual(smt_string: str) -> str:
    """
    Converts an SMT string to a MusicXML string manually, avoiding music21's object model pitfalls.
    """
    from datetime import date
    try:
        # --- XML Boilerplate ---
        xml_output = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<!DOCTYPE score-partwise  PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
            '<score-partwise version="4.0">',
        ]

        # --- Metadata ---
        title = "Untitled"
        subtitle = ""
        composer = "Unknown"
        
        parts = smt_string.split('|', 1)
        header_string = parts[0].strip()
        body_string = parts[1].strip() if len(parts) > 1 else ""
        header_tokens = header_string.split()

        time_signature = "4/4"
        clef_set = False

        for token_str in header_tokens:
            token = _parse_smt_token(token_str)
            if not token: continue
            if token["type"] == "title":
                title = token["value"]
            elif token["type"] == "subtitle":
                subtitle = token["value"]
            elif token["type"] in ["composer", "creator"]:
                composer = token["value"]
            elif token["type"] == "time":
                time_signature = token["value"]
            elif token["type"] == "clef" and token["value"] == "percussion":
                clef_set = True

        xml_output.append(f'  <work><work-title>{title}</work-title></work>')
        
        # Add credit tags
        xml_output.append(f'  <credit page="1"><credit-type>title</credit-type><credit-words justify="center" valign="top">{title}</credit-words></credit>')
        if subtitle:
            xml_output.append(f'  <credit page="1"><credit-type>subtitle</credit-type><credit-words justify="center" valign="top">{subtitle}</credit-words></credit>')
        xml_output.append(f'  <credit page="1"><credit-type>composer</credit-type><credit-words justify="right" valign="bottom">{composer}</credit-words></credit>')

        today = date.today().strftime("%Y-%m-%d")
        xml_output.append('  <identification>')
        xml_output.append(f'    <creator type="composer">{composer}</creator>')
        xml_output.append('    <encoding>')
        xml_output.append('      <software>RhythmForm</software>')
        xml_output.append(f'      <encoding-date>{today}</encoding-date>')
        xml_output.append('      <supports element="accidental" type="yes"/>')
        xml_output.append('      <supports element="beam" type="yes"/>')
        xml_output.append('      <supports element="print" attribute="new-page" type="yes" value="yes"/>')
        xml_output.append('      <supports element="print" attribute="new-system" type="yes" value="yes"/>')
        xml_output.append('      <supports element="stem" type="yes"/>')
        xml_output.append('    </encoding>')
        xml_output.append('  </identification>')
        
        # --- Part List ---
        xml_output.extend([
            '  <part-list>',
            '    <score-part id="P1">',
            f'      <part-name>{title}</part-name>',
            '    </score-part>',
            '  </part-list>',
            '  <!--=========================== Part 1 ===========================-->',
            '  <part id="P1">',
        ])

        # --- Process Measures ---
        measure_strings = body_string.split('|')
        measure_number = 1
        
        # MusicXML divisions - a high number for precision with fractions
        divisions = 10080 

        for i, measure_str in enumerate(measure_strings):
            xml_output.append(f'    <!--========================= Measure {measure_number} =========================-->')
            xml_output.append(f'    <measure number="{measure_number}">')

            # Add attributes (time signature, clef) to the first measure
            if i == 0:
                xml_output.append('      <attributes>')
                xml_output.append(f'        <divisions>{divisions}</divisions>')
                beats, beat_type = time_signature.split('/')
                xml_output.extend([
                    '        <time>',
                    f'          <beats>{beats}</beats>',
                    f'          <beat-type>{beat_type}</beat-type>',
                    '        </time>',
                ])
                if clef_set:
                    xml_output.extend([
                        '        <clef>',
                        '          <sign>percussion</sign>',
                        '        </clef>',
                    ])
                xml_output.append('      </attributes>')

            tokens = measure_str.strip().split()
            if not tokens:
                # Handle empty measures
                duration_in_beats = Fraction(beats) / Fraction(beat_type) * 4
                rest_duration_xml = int(duration_in_beats * divisions)
                xml_output.append(f'      <note><rest measure="yes"/><duration>{rest_duration_xml}</duration></note>')

            for token_str in tokens:
                token = _parse_smt_token(token_str)
                if not token: continue

                if token["type"] == "text":
                    xml_output.extend([
                        '      <direction placement="above">',
                        '        <direction-type><words>',
                        f'          {token["value"]}',
                        '        </words></direction-type>',
                        '      </direction>',
                    ])
                    continue

                duration_fraction = Fraction(token.get("duration", "0")).limit_denominator()
                xml_duration = int(duration_fraction * divisions)

                if token["type"] == "rest":
                    xml_output.append(f'      <note><rest/><duration>{xml_duration}</duration></note>')

                elif token["type"] == "note":
                    is_chord = len(token["instruments"]) > 1
                    
                    for i, instrument_name in enumerate(token["instruments"]):
                        if instrument_name not in SMT_TO_DRUM_DISPLAY:
                            continue # Skip unknown instruments
                        
                        display_info = SMT_TO_DRUM_DISPLAY[instrument_name]
                        
                        xml_output.append('      <note>')
                        if is_chord and i > 0:
                            xml_output.append('        <chord/>')
                        
                        xml_output.extend([
                            '        <unpitched>',
                            f'          <display-step>{display_info["display_step"]}</display-step>',
                            f'          <display-octave>{display_info["display_octave"]}</display-octave>',
                            '        </unpitched>',
                            f'        <duration>{xml_duration}</duration>',
                            f'        <notehead>{display_info["notehead"]}</notehead>',
                        ])
                        xml_output.append('      </note>')

            xml_output.append('    </measure>')
            measure_number += 1

        xml_output.extend(['  </part>', '</score-partwise>'])
        return "\n".join(xml_output)

    except Exception as e:
        import traceback
        print(f"Error processing SMT string manually: {e}")
        print(traceback.format_exc())
        return ""
