import random
import music21
import os
import json
import copy
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from glob import glob

import sys
# This block allows the script to be run from the command line (e.g. `python scripts/generate...`)
# by adding the project root to the Python path, so that imports like `from scripts...` work.
if __name__ == '__main__' and __package__ is None:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

# --- Markov Chain Imports ---
import pickle
from scripts.markov_chain.markov_chain import MarkovChain
from scripts.smt_to_musicxml import SmtConverter


# --- Configuration ---
# Uses the same environment variable and folder structure as prepare_dataset.py
TRAINING_DATA_DIR = Path(os.environ.get('RHYTHMFORMHOME', Path(__file__).parent.parent)) / 'training_data'
XML_OUTPUT_DIR = TRAINING_DATA_DIR / 'musicxml'


# --- Expanded drum instrument definition with notehead styles ---
# The tuple is (midi_number, display_step, display_octave, notehead_style)
DRUM_INSTRUMENTS = {
    'Bass Drum':    (36, 'F', 4, 'normal'),
    'Acoustic Snare':(38, 'C', 5, 'normal'),
    'Side Stick':   (37, 'C', 5, 'x'),
    'Closed Hi-Hat':(42, 'G', 5, 'x'),
    'Open Hi-Hat':  (46, 'G', 5, 'circle-x'),
    'Pedal Hi-Hat': (44, 'E', 4, 'x'),
    'Crash Cymbal': (49, 'A', 5, 'cross'),
    'Ride Cymbal':  (51, 'B', 5, 'cross'),
    'Low Tom':      (45, 'A', 4, 'normal'),
    'High Tom':     (50, 'E', 5, 'normal'),
}

# Possible note/rest durations (in quarter lengths)
DURATIONS = [0.25, 0.5, 1.0, 1.5, 2.0] # 16th, 8th, quarter, dotted 8th, half

# --- Possible part names for stave labels ---
PART_NAMES = ['Drumset', 'Drum Kit', 'Drums', 'Batterie', 'Schlagzeug']

def generate_drum_score(num_measures=16, output_path="synthetic_score.xml", complexity=0, use_repeats=False):
    """
    Generates a pseudo-random drum score and saves it as a MusicXML file.
    If use_repeats is True, it may mark measures as repeats in a companion JSON file.
    Complexity: 0=simple, 1=medium, 2=complex
    """
    # --- Define complexity levels ---
    if complexity == 0: # Simple
        active_instruments = {k: v for k, v in DRUM_INSTRUMENTS.items() if 'Hi-Hat' in k or 'Snare' in k or 'Bass Drum' in k}
        active_durations = [0.5, 1.0]
        chord_probability = 0.1
        repeat_probability = 0.4
    elif complexity == 1: # Medium
        active_instruments = {k: v for k, v in DRUM_INSTRUMENTS.items() if 'Ride' not in k}
        active_durations = [0.25, 0.5, 1.0]
        chord_probability = 0.3
        repeat_probability = 0.6
    else: # Complex
        active_instruments = DRUM_INSTRUMENTS
        active_durations = DURATIONS
        chord_probability = 0.5
        repeat_probability = 0.75

    # --- 1. Setup Score and Drum Part ---
    score = music21.stream.Score()
    drum_part = music21.stream.Part(id='drumset')
    
    # --- Randomly assign part name and abbreviation ---
    part_name = random.choice(PART_NAMES)
    drum_part.partName = part_name
    drum_part.partAbbreviation = part_name[:3] + '.'

    # --- THIS IS THE FIX: Define all instruments and add them to the part's instrument list ---
    instrument_definitions = {}
    # Clear any default instruments music21 might have added
    drum_part.instruments = [] 
    for name, (midi_num, _, _, _) in active_instruments.items():
        inst = music21.instrument.Percussion()
        inst.midiChannel = 10
        inst.midiUnpitched = midi_num
        inst_id = f"P{midi_num}"
        inst.instrumentId = inst_id
        instrument_definitions[midi_num] = inst
        # Add the instrument to the part's list of instruments.
        # This is the correct way to ensure they appear in the <part-list>.
        drum_part.instruments.append(inst)

    drum_part.insert(0, music21.clef.PercussionClef())
    drum_part.insert(0, music21.meter.TimeSignature('4/4'))

    # --- 2. Generate Measures ---
    previous_measure_elements = None
    repeated_measure_numbers = []

    for i in range(num_measures):
        measure = music21.stream.Measure(number=i + 1)
        
        should_repeat = (
            use_repeats and
            previous_measure_elements is not None and
            i > 0 and
            random.random() < repeat_probability
        )

        if should_repeat:
            # --- THIS IS THE FIX ---
            # We must create deep copies of the elements from the previous measure.
            # The correct way to do this for music21 objects is with copy.deepcopy().
            current_measure_elements = []
            if previous_measure_elements is not None:
                for el in previous_measure_elements:
                    el_copy = copy.deepcopy(el) # Deep copy using the copy module
                    measure.insert(el.offset, el_copy)
                    current_measure_elements.append(el_copy)
            
            # Mark this measure number for later modification in prepare_dataset.py
            repeated_measure_numbers.append(i + 1)
            # The new measure's elements become the basis for the next potential repeat
            previous_measure_elements = current_measure_elements
        else:
            # Generate a new, unique measure
            current_offset = 0.0
            current_measure_elements = []
            while current_offset < 4.0:
                remaining_duration = 4.0 - current_offset
                possible_durations = [d for d in active_durations if d <= remaining_duration]
                if not possible_durations:
                    rest = music21.note.Rest(quarterLength=remaining_duration)
                    measure.insert(current_offset, rest)
                    current_measure_elements.append(rest)
                    break
                
                duration = random.choice(possible_durations)

                # Generate note or rest
                if random.random() < 0.85: # 85% chance of a note/chord
                    is_chord = random.random() < chord_probability
                    note_event = music21.percussion.PercussionChord() if is_chord else music21.note.Unpitched()
                    num_notes_in_event = random.randint(2, 3) if is_chord else 1
                    
                    # Ensure unique instruments in a chord
                    instruments_in_chord = random.sample(list(active_instruments.keys()), num_notes_in_event)

                    for instrument_name in instruments_in_chord:
                        midi_num, d_step, d_octave, notehead_style = active_instruments[instrument_name]
                        
                        unpitched_note = music21.note.Unpitched()
                        unpitched_note.displayStep = d_step
                        unpitched_note.displayOctave = d_octave
                        unpitched_note.notehead = notehead_style
                        
                        # --- THIS IS THE FIX ---
                        # Refer to the predefined instrument object by its ID.
                        # music21 will create the correct <instrument id="..."> tag.
                        unpitched_note.instrument = instrument_definitions[midi_num]
                        
                        if is_chord:
                            note_event.add(unpitched_note)
                        else:
                            note_event = unpitched_note
                            
                    note_event.duration.quarterLength = duration
                    measure.insert(current_offset, note_event)
                    current_measure_elements.append(note_event)
                else: # 15% chance of a rest
                    rest = music21.note.Rest(quarterLength=duration)
                    measure.insert(current_offset, rest)
                    current_measure_elements.append(rest)
                
                current_offset += duration
            
            # Store the elements of this new measure in case the next one is a repeat
            previous_measure_elements = current_measure_elements

        # Set barlines. The last measure gets a final barline.
        if i == num_measures - 1:
            measure.rightBarline = 'final'
        else:
            measure.rightBarline = 'regular'

        drum_part.append(measure)

    score.insert(0, drum_part)
    
    # --- 3. Save Files ---
    # Save the MusicXML file with all notes explicitly written
    score.write('musicxml', fp=output_path)
    print(f"Successfully generated randomised score at: {output_path}")

    # Save the companion JSON file with the list of repeated measure numbers
    if repeated_measure_numbers:
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({"repeated_measures": repeated_measure_numbers}, f)
        print(f"  -> Found repeats, saved info to: {json_path}")


def generate_markov_score(markov_model, output_path, complexity=0, title="Synthetic Score"):
    """
    Generates a score using a trained MarkovChain model, ensuring valid measure durations.
    """
    from fractions import Fraction

    # Adjust generation length based on complexity
    if complexity == 0:
        num_measures = random.randint(12, 18)
    elif complexity == 1:
        num_measures = random.randint(16, 24)
    else:
        num_measures = random.randint(20, 32)
    
    time_signature = Fraction(4, 4)
    
    generated_measures = []
    
    for _ in range(num_measures):
        current_measure_tokens = []
        current_duration = Fraction(0)
        
        while current_duration < time_signature:
            remaining_duration = time_signature - current_duration
            
            # Generate a token
            # We can pass the last token to guide the generation
            start_token = current_measure_tokens[-1] if current_measure_tokens else None
            token = markov_model.generate(length=1, start_token=start_token)[0]

            # Extract duration from token
            try:
                # Basic parsing to find duration like "note[...,1/4]" or "rest[1/2]"
                content = token.split('[')[1].split(']')[0]
                if "note" in token:
                    token_duration = Fraction(content.split(',')[-1])
                elif "rest" in token:
                    token_duration = Fraction(content)
                else: # e.g. text[...]
                    token_duration = Fraction(0)
            except (IndexError, ValueError):
                # If token has no duration or is malformed, skip it
                continue
            
            # If the token fits, add it. Otherwise, fill with a rest and break.
            if current_duration + token_duration <= time_signature:
                current_measure_tokens.append(token)
                current_duration += token_duration
            else:
                # Fill the rest of the measure with a rest
                fill_rest_duration = remaining_duration
                if fill_rest_duration > 0:
                    current_measure_tokens.append(f"rest[{fill_rest_duration}]")
                current_duration += fill_rest_duration # This will end the loop
        
        generated_measures.append(" ".join(current_measure_tokens))

    # Join measures with barlines
    body = " | ".join(generated_measures)
    
    # Add required metadata
    metadata = [
        f"title[{title}]",
        "subtitle[Generated by RhythmForm Markov Model]",
        "composer[RhythmForm]",
        "clef[percussion]",
        "time[4/4]"
    ]
    full_sequence_str = " ".join(metadata) + " | " + body

    # Convert the SMT string to a MusicXML file
    converter = SmtConverter(full_sequence_str)
    if converter.write_musicxml(output_path):
        print(f"Successfully generated Markov score at: {output_path}")
    else:
        print(f"Failed to generate Markov score at: {output_path}")


if __name__ == '__main__':
    XML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Generate synthetic drum scores.")
    parser.add_argument(
        "num_scores", 
        type=int, 
        nargs='?', 
        default=30, 
        help="Number of scores to generate (default: 30)"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to use for parallel generation (default: 1)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for score generation (default: 0)"
    )
    parser.add_argument(
        "--markov-model",
        type=str,
        help="Path to the trained MarkovChain model (pickle file)."
    )
    parser.add_argument(
        "--markov-ratio",
        type=float,
        default=0.8,
        help="Ratio of scores to generate using the Markov model (default: 0.8)"
    )
    args = parser.parse_args()
    
    num_scores_to_generate = args.num_scores
    num_cores_to_use = args.cores
    start_index = args.start_index
    print(f"Generating {num_scores_to_generate} scores into {XML_OUTPUT_DIR} using {num_cores_to_use} cores, starting at index {start_index}")

    # --- Load Markov Model if specified ---
    markov_model = None
    if args.markov_ratio > 0:
        if not args.markov_model:
            raise ValueError("--markov-model must be specified when --markov-ratio > 0")
        with open(args.markov_model, 'rb') as f:
            markov_model = pickle.load(f)
        print(f"Loaded Markov model from {args.markov_model}")


    tasks = []
    # This logic is now simplified. We just generate scores starting from the given index.
    # The complexity level can be based on the global index.
    with ProcessPoolExecutor(max_workers=num_cores_to_use) as executor:
        for i in range(num_scores_to_generate):
            score_index = i + start_index
            
            # Determine complexity based on the overall score index
            if score_index < 40000: # Example threshold
                level = 0
            elif score_index < 80000: # Example threshold
                level = 1
            else:
                level = 2

            use_repeats_for_this_score = random.random() < 0.5
            file_path = XML_OUTPUT_DIR / f"synthetic_score_{score_index + 1}_level_{level}.xml"

            # Decide whether to use Markov Chain or random generation based on the ratio
            if markov_model and random.random() < args.markov_ratio:
                # Submit a Markov generation task
                tasks.append(
                    executor.submit(
                        generate_markov_score,
                        markov_model,
                        output_path=file_path,
                        complexity=level,
                        title=f"Synthetic Score {score_index + 1}"
                    )
                )
            else:
                # Submit a random generation task
                tasks.append(
                    executor.submit(
                        generate_drum_score,
                        num_measures=random.randint(12, 24),
                        output_path=file_path,
                        complexity=level,
                        use_repeats=use_repeats_for_this_score
                    )
                )

        # Optionally, show progress
        for tqdm_instance in tqdm(as_completed(tasks), total=len(tasks)):
            pass

    print("All synthetic scores generated.")