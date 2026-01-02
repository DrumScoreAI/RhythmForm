import music21
from fractions import Fraction
from bs4 import BeautifulSoup

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


def _get_duration_map_from_xml(xml_path):
    """
    Parses the MusicXML to create a mapping from measure and element index
    to its correct fractional duration. This avoids music21's parsing bugs.
    The key is a tuple (measure_number, element_index_in_measure), and the
    value is the duration as a Fraction.
    """
    duration_map = {}
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')

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
        print(f"Warning: Could not build duration map from {xml_path}: {e}")
        
    return duration_map


def _extract_instrument_map_from_xml(xml_path):
    """
    Directly parses the MusicXML file using BeautifulSoup to find <score-instrument>
    definitions and create a mapping from instrument ID to MIDI number. This is a
    robust way to get this information, bypassing music21's parsing inconsistencies.
    """
    id_to_midi_map = {}
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')

        for instrument in soup.find_all('score-instrument'):
            inst_id = instrument.get('id')
            midi_unpitched_tag = instrument.find('midi-unpitched')
            
            if inst_id and midi_unpitched_tag and midi_unpitched_tag.string:
                try:
                    midi_num = int(midi_unpitched_tag.string)
                    id_to_midi_map[inst_id] = midi_num
                except (ValueError, TypeError):
                    continue # Ignore if the midi number is not a valid integer
                    
    except Exception as e:
        print(f"Warning: Could not parse instrument map from {xml_path} with BeautifulSoup: {e}")
    
    return id_to_midi_map


def _convert_note_to_smt(element, duration, id_to_midi_map=None):
    """
    Converts a single music21 element (Note, Rest, Chord) to its SMT string representation.
    Internal helper function for musicxml_to_smt.
    """
    if id_to_midi_map is None:
        id_to_midi_map = {}

    if isinstance(element, music21.note.Rest):
        return f"rest[{duration}]"

    
    all_notes = []
    if isinstance(element, music21.note.Note):
        all_notes.append(element)
    elif isinstance(element, music21.chord.ChordBase):
        # For PercussionChord, notes are in the notes attribute
        all_notes.extend(element.notes if hasattr(element, 'notes') else element)

    smt_notes = []
    for note_obj in all_notes:
        # Fallback for unpitched notes using display properties
        if isinstance(note_obj, music21.note.Unpitched):
            notehead = note_obj.notehead if note_obj.notehead else 'normal'
            display_key = (note_obj.displayStep, note_obj.displayOctave, notehead)
            if display_key in DRUM_DISPLAY_TO_SMT:
                smt_notes.append(DRUM_DISPLAY_TO_SMT[display_key])
                continue

        # Try to get SMT from instrument MIDI mapping first
        instrument = note_obj.getInstrument()
        if instrument and instrument.instrumentId in id_to_midi_map:
            midi_num = id_to_midi_map[instrument.instrumentId]
            if midi_num in DRUM_MIDI_TO_SMT:
                smt_notes.append(DRUM_MIDI_TO_SMT[midi_num])
                continue

        # Fallback to MIDI pitch number for pitched notes
        if hasattr(note_obj, 'pitch') and note_obj.pitch.midi in DRUM_MIDI_TO_SMT:
            smt_notes.append(DRUM_MIDI_TO_SMT[note_obj.pitch.midi])
            continue

    if not smt_notes:
        return None # No valid SMT mapping found

    # Join multiple notes in a chord with a dot
    note_str = ".".join(sorted(list(set(smt_notes))))
    return f"note[{note_str},{duration}]"


def _measure_to_smt(measure, duration_map, is_repeated=False, id_to_midi_map=None):
    """
    Converts a single music21 measure to its SMT representation.
    If is_repeated is True, it returns a special repeat token.
    """
    if id_to_midi_map is None:
        id_to_midi_map = {}

    if is_repeated:
        return "repeat[measure]"

    smt_tokens = []
    if measure:
        element_idx = 0
        for element in measure.notesAndRests:
            # Retrieve the correct duration from our pre-parsed map
            duration = duration_map.get((measure.number, element_idx), Fraction(element.duration.quarterLength).limit_denominator())
            
            token = _convert_note_to_smt(element, duration, id_to_midi_map)
            if token:
                smt_tokens.append(token)
            
            # Only increment for non-chord elements
            if not (isinstance(element, music21.note.Note) and element.isChord):
                element_idx += 1
            
    return " ".join(smt_tokens)


def musicxml_to_smt(xml_path):
    """
    Converts a MusicXML file to its SMT representation.
    This is the main entry point for MusicXML to SMT conversion.
    """
    try:
        # Use music21 to parse the score. It's good at handling the overall structure.
        score = music21.converter.parse(xml_path)
    except Exception as e:
        print(f"Error parsing MusicXML file with music21: {e}")
        return ""

    # Directly parse the XML to get accurate duration and instrument info,
    # bypassing potential music21 parsing bugs.
    duration_map = _get_duration_map_from_xml(xml_path)
    id_to_midi_map = _extract_instrument_map_from_xml(xml_path)

    # Get the first instrument part (assuming single-instrument scores for now)
    part = score.parts[0] if score.parts else None
    if not part:
        return ""

    smt_measures = []
    for measure in part.getElementsByClass('Measure'):
        # Check for repeat signs. music21 can identify repeats.
        is_repeated = False
        if measure.leftBarline and isinstance(measure.leftBarline, music21.bar.Repeat) and measure.leftBarline.direction == 'end':
            is_repeated = True
        
        measure_smt = _measure_to_smt(measure, duration_map, is_repeated=is_repeated, id_to_midi_map=id_to_midi_map)
        if measure_smt:
            smt_measures.append(measure_smt)

    return " | ".join(smt_measures)


