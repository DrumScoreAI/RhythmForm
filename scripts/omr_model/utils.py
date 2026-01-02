import music21
from fractions import Fraction
import re

import music21
from fractions import Fraction
from bs4 import BeautifulSoup

# This file contains shared utility functions for the OMR model,
# primarily for converting between MusicXML and Symbolic Music Text (SMT) formats.
# ...
# ('E', 5, 'normal'): 'HT',
}


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


def _convert_note_to_smt(element, id_to_midi_map=None):
    """
    Converts a single music21 element (Note, Rest, Chord) to its SMT string representation.
    Internal helper function for musicxml_to_smt.
    """
    if id_to_midi_map is None:
        id_to_midi_map = {}

    if isinstance(element, music21.note.Rest):
        duration = Fraction(element.duration.quarterLength).limit_denominator()
        return f"rest[{duration}]"

    duration = Fraction(element.duration.quarterLength).limit_denominator()
    
    all_notes = []
    if isinstance(element, music21.note.Note):
        all_notes.append(element)
    elif isinstance(element, music21.chord.ChordBase):
        # For PercussionChord, notes are in the notes attribute
        all_notes.extend(element.notes if hasattr(element, 'notes') else element)


    instrument_names = set()
    for n in all_notes:
        smt_name = None
        midi_pitch = None
        
        # --- THIS IS THE FIX ---
        # The logic is reordered to be more robust.
        # 1. Prioritize the instrument ID link, which is the most reliable method.
        # 2. Fallback to other methods for compatibility.
        inst = n.getInstrument() # FIX: Use the getInstrument() method to retrieve the assigned instrument.
        if inst and hasattr(inst, 'instrumentId') and inst.instrumentId in id_to_midi_map:
            # Preferred method: The note is explicitly linked to an instrument definition
            # via an ID. This works with the new, correct MusicXML generation.
            midi_pitch = id_to_midi_map.get(inst.instrumentId)
            if midi_pitch:
                smt_name = DRUM_MIDI_TO_SMT.get(midi_pitch)
        
        if not smt_name and hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
            # Fallback 1: Standard case for a note with a MIDI pitch.
            midi_pitch = n.pitch.midi
            smt_name = DRUM_MIDI_TO_SMT.get(midi_pitch)

        if not smt_name and isinstance(n, music21.note.Unpitched):
            # Fallback 2: Case for <unpitched> notes where we guess from visual properties.
            # This is less reliable but useful for some generated scores.
            notehead = n.notehead if n.notehead else 'normal'
            display_step = n.displayStep if hasattr(n, 'displayStep') else None
            if display_step:
                # Create a more specific key to avoid collisions
                display_key = (display_step, n.displayOctave, notehead)
                smt_name = DRUM_DISPLAY_TO_SMT.get(display_key)

        if not smt_name and inst and hasattr(inst, 'midiChannel') and inst.midiChannel == 10 and hasattr(inst, 'midiUnpitched'):
            # Fallback 3: Case for old-style files where the instrument was embedded in the note.
            midi_pitch = inst.midiUnpitched
            smt_name = DRUM_MIDI_TO_SMT.get(midi_pitch)
        
        if smt_name:
            instrument_names.add(smt_name)
    
    if not instrument_names:
        # If after all checks, we still have nothing, log it for debugging.
        # Return None so it doesn't create an empty 'note[]' token.
        # print(f"DEBUG: Could not determine instrument for element: {element} with notes {all_notes}")
        return None

    sorted_names = sorted(list(instrument_names))
    return f"note[{'&'.join(sorted_names)},{duration}]"


def _measure_to_smt(measure, is_repeated=False, id_to_midi_map=None):
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
        for element in measure.notesAndRests:
            token = _convert_note_to_smt(element, id_to_midi_map)
            if token:
                smt_tokens.append(token)
            
    return " ".join(smt_tokens)


def musicxml_to_smt(score_path, repeated_measures=None):
    """
    Converts a full MusicXML file to a single-line SMT string.
    - Manually parses the XML to get an accurate instrument map.
    - Parses the file with music21 to get the musical structure.
    - Extracts the drum part.
    - Iterates through all elements and converts them to SMT.
    - Joins tokens with 'measure_break'.
    """
    if repeated_measures is None:
        repeated_measures = []

    # --- THIS IS THE DEFINITIVE FIX ---
    # 1. Manually parse the XML to build the instrument map.
    id_to_midi_map = _extract_instrument_map_from_xml(score_path)

    try:
        # 2. Now, parse the score with music21 to get the structure.
        score = music21.converter.parse(score_path)
        drum_part = None
        
        # Find the drum part (same logic as before)
        for part in score.parts:
            if part.getElementsByClass('PercussionClef'):
                drum_part = part
                break
        if not drum_part:
            for part in score.parts:
                instrument = part.getInstrument()
                if instrument and any(s in (instrument.instrumentSound or '').lower() for s in ['percussion', 'drum']) or any(s in (instrument.instrumentName or '').lower() for s in ['batterie', 'schlagzeug']):
                    drum_part = part
                    break
        if not drum_part:
            drum_part = score.parts[0] if score.parts else None

        if not drum_part:
            print(f"Error: No parts found in {score_path}")
            return None
        
        smt_components = []
        
        # Process metadata that appears before the first measure
        clef = drum_part.getElementsByClass('Clef').first()
        if clef and isinstance(clef, music21.clef.PercussionClef):
            smt_components.append("clef[percussion]")
            
        ts = drum_part.getElementsByClass('TimeSignature').first()
        if ts:
            smt_components.append(f"timeSignature[{ts.numerator}/{ts.denominator}]")

        # Process measures
        for measure in drum_part.getElementsByClass('Measure'):
            is_repeat = measure.number in repeated_measures
            measure_smt = _measure_to_smt(measure, is_repeated=is_repeat, id_to_midi_map=id_to_midi_map)
            if measure_smt:
                smt_components.append(measure_smt)
        
        final_smt = " measure_break ".join(smt_components)
        return " ".join(final_smt.strip().split())

    except Exception as e:
        print(f"Error parsing {score_path} with music21: {e}")
        import traceback
        traceback.print_exc()
        return None


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


def _extract_instrument_map_from_xml(xml_path):
    """
    Directly parses the MusicXML file to find <score-instrument> definitions
    and create a mapping from instrument ID to MIDI number. This is a robust
    way to get this information, bypassing music21's inconsistent parsing of it.
    """
    id_to_midi_map = {}
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Regex to find all score-instrument blocks
        instrument_blocks = re.findall(r'<score-instrument id="([^"]+)">(.+?)</score-instrument>', xml_content, re.DOTALL)

        for inst_id, block_content in instrument_blocks:
            # Regex to find the midi-unpitched value within the block
            midi_match = re.search(r'<midi-unpitched>(\d+)</midi-unpitched>', block_content)
            if midi_match:
                midi_num = int(midi_match.group(1))
                id_to_midi_map[inst_id] = midi_num
    except Exception as e:
        print(f"Warning: Could not manually parse instrument map from {xml_path}: {e}")
    
    print(f"DEBUG: Final Instrument Map from XML: {id_to_midi_map}")
    return id_to_midi_map


def _convert_note_to_smt(element, id_to_midi_map=None):
    """
    Converts a single music21 element (Note, Rest, Chord) to its SMT string representation.
    Internal helper function for musicxml_to_smt.
    """
    if id_to_midi_map is None:
        id_to_midi_map = {}

    if isinstance(element, music21.note.Rest):
        duration = Fraction(element.duration.quarterLength).limit_denominator()
        return f"rest[{duration}]"

    duration = Fraction(element.duration.quarterLength).limit_denominator()
    
    all_notes = []
    if isinstance(element, music21.note.Note):
        all_notes.append(element)
    elif isinstance(element, music21.chord.ChordBase):
        # For PercussionChord, notes are in the notes attribute
        all_notes.extend(element.notes if hasattr(element, 'notes') else element)


    instrument_names = set()
    for n in all_notes:
        smt_name = None
        midi_pitch = None
        
        # --- THIS IS THE FIX ---
        # The logic is reordered to be more robust.
        # 1. Prioritize the instrument ID link, which is the most reliable method.
        # 2. Fallback to other methods for compatibility.
        inst = n.getInstrument() # FIX: Use the getInstrument() method to retrieve the assigned instrument.
        if inst and hasattr(inst, 'instrumentId') and inst.instrumentId in id_to_midi_map:
            # Preferred method: The note is explicitly linked to an instrument definition
            # via an ID. This works with the new, correct MusicXML generation.
            midi_pitch = id_to_midi_map.get(inst.instrumentId)
            if midi_pitch:
                smt_name = DRUM_MIDI_TO_SMT.get(midi_pitch)
        
        if not smt_name and hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
            # Fallback 1: Standard case for a note with a MIDI pitch.
            midi_pitch = n.pitch.midi
            smt_name = DRUM_MIDI_TO_SMT.get(midi_pitch)

        if not smt_name and isinstance(n, music21.note.Unpitched):
            # Fallback 2: Case for <unpitched> notes where we guess from visual properties.
            # This is less reliable but useful for some generated scores.
            notehead = n.notehead if n.notehead else 'normal'
            display_step = n.displayStep if hasattr(n, 'displayStep') else None
            if display_step:
                # Create a more specific key to avoid collisions
                display_key = (display_step, n.displayOctave, notehead)
                smt_name = DRUM_DISPLAY_TO_SMT.get(display_key)

        if not smt_name and inst and hasattr(inst, 'midiChannel') and inst.midiChannel == 10 and hasattr(inst, 'midiUnpitched'):
            # Fallback 3: Case for old-style files where the instrument was embedded in the note.
            midi_pitch = inst.midiUnpitched
            smt_name = DRUM_MIDI_TO_SMT.get(midi_pitch)
        
        if smt_name:
            instrument_names.add(smt_name)
    
    if not instrument_names:
        # If after all checks, we still have nothing, log it for debugging.
        # Return None so it doesn't create an empty 'note[]' token.
        # print(f"DEBUG: Could not determine instrument for element: {element} with notes {all_notes}")
        return None

    sorted_names = sorted(list(instrument_names))
    return f"note[{'&'.join(sorted_names)},{duration}]"


def _measure_to_smt(measure, is_repeated=False, id_to_midi_map=None):
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
        for element in measure.notesAndRests:
            token = _convert_note_to_smt(element, id_to_midi_map)
            if token:
                smt_tokens.append(token)
            
    return " ".join(smt_tokens)


def musicxml_to_smt(score_path, repeated_measures=None):
    """
    Converts a full MusicXML file to a single-line SMT string.
    - Manually parses the XML to get an accurate instrument map.
    - Parses the file with music21 to get the musical structure.
    - Extracts the drum part.
    - Iterates through all elements and converts them to SMT.
    - Joins tokens with 'measure_break'.
    """
    if repeated_measures is None:
        repeated_measures = []

    # --- THIS IS THE DEFINITIVE FIX ---
    # 1. Manually parse the XML to build the instrument map.
    id_to_midi_map = _extract_instrument_map_from_xml(score_path)

    try:
        # 2. Now, parse the score with music21 to get the structure.
        score = music21.converter.parse(score_path)
        drum_part = None
        
        # Find the drum part (same logic as before)
        for part in score.parts:
            if part.getElementsByClass('PercussionClef'):
                drum_part = part
                break
        if not drum_part:
            for part in score.parts:
                instrument = part.getInstrument()
                if instrument and any(s in (instrument.instrumentSound or '').lower() for s in ['percussion', 'drum']) or any(s in (instrument.instrumentName or '').lower() for s in ['batterie', 'schlagzeug']):
                    drum_part = part
                    break
        if not drum_part:
            drum_part = score.parts[0] if score.parts else None

        if not drum_part:
            print(f"Error: No parts found in {score_path}")
            return None
        
        smt_components = []
        
        # Process metadata that appears before the first measure
        clef = drum_part.getElementsByClass('Clef').first()
        if clef and isinstance(clef, music21.clef.PercussionClef):
            smt_components.append("clef[percussion]")
            
        ts = drum_part.getElementsByClass('TimeSignature').first()
        if ts:
            smt_components.append(f"timeSignature[{ts.numerator}/{ts.denominator}]")

        # Process measures
        for measure in drum_part.getElementsByClass('Measure'):
            is_repeat = measure.number in repeated_measures
            measure_smt = _measure_to_smt(measure, is_repeated=is_repeat, id_to_midi_map=id_to_midi_map)
            if measure_smt:
                smt_components.append(measure_smt)
        
        final_smt = " measure_break ".join(smt_components)
        return " ".join(final_smt.strip().split())

    except Exception as e:
        print(f"Error parsing {score_path} with music21: {e}")
        import traceback
        traceback.print_exc()
        return None
