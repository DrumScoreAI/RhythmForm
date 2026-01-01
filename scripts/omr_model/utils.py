import music21
from fractions import Fraction

# This file contains shared utility functions for the OMR model,
# primarily for converting between MusicXML and Symbolic Music Text (SMT) formats.

# --- Drum MIDI to SMT Mapping ---
# This dictionary maps MIDI numbers to the SMT representation.
# It's consolidated here to be the single source of truth.
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

def _convert_note_to_smt(element):
    """
    Converts a single music21 element (Note, Rest, Chord) to its SMT string representation.
    Internal helper function for musicxml_to_smt.
    """
    if isinstance(element, music21.note.Rest):
        duration = Fraction(element.duration.quarterLength).limit_denominator()
        return f"rest[{duration}]"

    duration = Fraction(element.duration.quarterLength).limit_denominator()
    
    all_notes = []
    if isinstance(element, music21.note.Note):
        all_notes.append(element)
    elif isinstance(element, music21.chord.ChordBase):
        all_notes.extend(element.notes)

    instrument_names = set()
    for n in all_notes:
        # In music21, percussion notes are often regular notes with MIDI pitches.
        if hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
            smt_name = DRUM_MIDI_TO_SMT.get(n.pitch.midi)
            if smt_name:
                instrument_names.add(smt_name)
    
    if not instrument_names:
        return None

    sorted_names = sorted(list(instrument_names))
    return f"note[{'&'.join(sorted_names)},{duration}]"


def _measure_to_smt(measure, is_repeated=False):
    """
    Converts a single music21 measure to its SMT representation.
    If is_repeated is True, it returns a special repeat token.
    """
    if is_repeated:
        return "repeat[measure]"

    # Group elements by offset to correctly handle chords
    elements_by_offset = {}
    for el in measure.notesAndRests:
        if el.offset not in elements_by_offset:
            elements_by_offset[el.offset] = []
        elements_by_offset[el.offset].append(el)

    smt_tokens = []
    for offset in sorted(elements_by_offset.keys()):
        elements = elements_by_offset[offset]
        
        # For chords, music21 might represent them as multiple notes at the same offset
        # or as a Chord object. We'll treat them as a single event.
        combined_element = elements[0]
        if len(elements) > 1:
            # Create a chord from all notes at this offset for easier processing
            combined_element = music21.chord.Chord(elements)

        token = _convert_note_to_smt(combined_element)
        if token:
            smt_tokens.append(token)
            
    return " ".join(smt_tokens)


def musicxml_to_smt(score_path, repeated_measures=None):
    """
    Converts a full MusicXML file to a single-line SMT string.
    - Parses a MusicXML file into a music21 stream.
    - Extracts the drum part.
    - Converts each measure to SMT, handling repeats if specified.
    - Joins measures with 'measure_break'.
    """
    if repeated_measures is None:
        repeated_measures = []

    try:
        score = music21.converter.parse(score_path)
        drum_part = None
        for part in score.parts:
            # A more version-agnostic way to find the percussion part.
            # We check the instrument's sound name.
            instrument = part.getInstrument()
            if instrument and ('percussion' in instrument.instrumentSound or 'drum' in instrument.instrumentSound):
                drum_part = part
                break
        
        if not drum_part:
            drum_part = score.parts[0] # Fallback to the first part

        full_smt = []
        
        # Add initial time signature from the first measure
        first_measure = drum_part.measure(1)
        if first_measure and first_measure.timeSignature:
            ts = first_measure.timeSignature
            full_smt.append(f"timeSignature[{ts.numerator}/{ts.denominator}]")

        for measure in drum_part.getElementsByClass('Measure'):
            is_repeat = measure.number in repeated_measures
            smt_measure_tokens = _measure_to_smt(measure, is_repeated=is_repeat)
            if smt_measure_tokens: # Only add if the measure wasn't empty
                full_smt.append(smt_measure_tokens)
        
        final_smt = " measure_break ".join(full_smt)
        return " ".join(final_smt.strip().split())

    except Exception as e:
        print(f"Error parsing {score_path} with music21: {e}")
        return None
