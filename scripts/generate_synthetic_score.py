import random
import music21
import os
from pathlib import Path

# --- Configuration ---
# Uses the same environment variable and folder structure as prepare_dataset.py
TRAINING_DATA_DIR = Path(os.environ.get('SFHOME', Path(__file__).parent.parent)) / 'training_data'
XML_OUTPUT_DIR = TRAINING_DATA_DIR / 'musicxml'


# --- Expanded drum instrument definition with notehead styles ---
# The tuple is (midi_number, staff_line_position, notehead_style)
DRUM_INSTRUMENTS = {
    'Bass Drum':    (36, -1, 'normal'),
    'Acoustic Snare':(38, 2, 'normal'),
    'Side Stick':   (37, 2, 'x'),
    'Closed Hi-Hat':(42, 4, 'x'),
    'Open Hi-Hat':  (46, 4, 'circle-x'),
    'Pedal Hi-Hat': (44, -2, 'x'),
    'Crash Cymbal': (49, 5, 'cross'),
    'Ride Cymbal':  (51, 6, 'cross'),
    'Low Tom':      (45, 1, 'normal'),
    'High Tom':     (50, 3, 'normal'),
}

# Possible note/rest durations (in quarter lengths)
DURATIONS = [0.25, 0.5, 1.0, 1.5, 2.0] # 16th, 8th, quarter, dotted 8th, half

# --- Possible part names for stave labels ---
PART_NAMES = ['Drumset', 'Drum Kit', 'Drums', 'Batterie', 'Schlagzeug']

def generate_drum_score(num_measures=16, output_path="synthetic_score.xml", complexity=0):
    """
    Generates a pseudo-random drum score and saves it as a MusicXML file.
    Complexity: 0=simple, 1=medium, 2=complex
    """
    # --- Define complexity levels ---
    if complexity == 0: # Simple
        active_instruments = {k: v for k, v in DRUM_INSTRUMENTS.items() if 'Hi-Hat' in k or 'Snare' in k or 'Bass Drum' in k}
        active_durations = [0.5, 1.0]
        chord_probability = 0.1
    elif complexity == 1: # Medium
        active_instruments = {k: v for k, v in DRUM_INSTRUMENTS.items() if 'Ride' not in k}
        active_durations = [0.25, 0.5, 1.0]
        chord_probability = 0.3
    else: # Complex
        active_instruments = DRUM_INSTRUMENTS
        active_durations = DURATIONS
        chord_probability = 0.5

    # --- 1. Setup Score and Drum Part ---
    score = music21.stream.Score()
    drum_part = music21.stream.Part(id='drumset')
    
    # --- Randomly assign part name and abbreviation ---
    part_name = random.choice(PART_NAMES)
    drum_part.partName = part_name
    drum_part.partAbbreviation = part_name[:3] + '.'
    
    drum_part.insert(0, music21.instrument.Percussion())
    drum_part.insert(0, music21.clef.PercussionClef())
    drum_part.insert(0, music21.meter.TimeSignature('4/4'))

    # --- 2. Generate Measures ---
    for i in range(num_measures):
        measure = music21.stream.Measure(number=i + 1)
        
        current_offset = 0.0
        while current_offset < 4.0:
            remaining_duration = 4.0 - current_offset
            possible_durations = [d for d in active_durations if d <= remaining_duration]
            if not possible_durations:
                rest = music21.note.Rest(quarterLength=remaining_duration)
                measure.insert(current_offset, rest)
                break
            
            duration = random.choice(possible_durations)

            if random.random() < 0.85:
                is_chord = random.random() < chord_probability
                note_event = music21.percussion.PercussionChord() if is_chord else music21.note.Unpitched()
                num_notes_in_event = 2 if is_chord else 1
                
                for _ in range(num_notes_in_event):
                    instrument_name = random.choice(list(active_instruments.keys()))
                    midi_num, staff_pos, notehead_style = active_instruments[instrument_name]
                    
                    unpitched_note = music21.note.Unpitched()
                    p = music21.pitch.Pitch(midi=midi_num)
                    unpitched_note.displayStep = p.step
                    unpitched_note.displayOctave = p.octave + 2
                    
                    unpitched_note.notehead = notehead_style
                    
                    if is_chord:
                        note_event.add(unpitched_note)
                    else:
                        note_event = unpitched_note
                        
                note_event.duration.quarterLength = duration
                measure.insert(current_offset, note_event)
            else:
                rest = music21.note.Rest(quarterLength=duration)
                measure.insert(current_offset, rest)
            
            current_offset += duration
            
        drum_part.append(measure)

    score.insert(0, drum_part)
    
    # --- 3. Save File ---
    score.write('musicxml', fp=output_path)
    print(f"Successfully generated synthetic score at: {output_path}")


if __name__ == '__main__':
    XML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_scores_to_generate = 30
    print(f"Generating {num_scores_to_generate} scores into {XML_OUTPUT_DIR}")
    
    for i in range(num_scores_to_generate):
        if i < num_scores_to_generate / 3:
            level = 0
        elif i < num_scores_to_generate * 2 / 3:
            level = 1
        else:
            level = 2

        file_path = XML_OUTPUT_DIR / f"synthetic_score_{i+1}_level_{level}.xml"
        generate_drum_score(
            num_measures=random.randint(12, 24), 
            output_path=file_path,
            complexity=level
        )
