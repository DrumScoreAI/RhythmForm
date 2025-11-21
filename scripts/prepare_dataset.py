import os
from pathlib import Path
import csv
import json
import music21
from pdf2image import convert_from_path
from fractions import Fraction

# --- Configuration ---
TRAINING_DATA_DIR = Path(os.environ.get('SFHOME', Path(__file__).parent.parent)) / 'training_data'
PDF_DIR = TRAINING_DATA_DIR / 'pdfs'
XML_DIR = TRAINING_DATA_DIR / 'musicxml'
OUTPUT_IMAGE_DIR = TRAINING_DATA_DIR / 'images'
MANIFEST_FILE = TRAINING_DATA_DIR / 'training_data.csv'
FINAL_DATASET_FILE = TRAINING_DATA_DIR / 'dataset.json'

# --- Drum MIDI to SMT Mapping ---
# This dictionary maps MIDI numbers to the SMT representation.
DRUM_MIDI_TO_SMT = {
    # Bass Drums
    35: 'BD',  # Acoustic Bass Drum
    36: 'BD',  # Bass Drum 1
    
    # Snares
    38: 'SD',  # Acoustic Snare
    40: 'SD',  # Electric Snare
    37: 'SS',  # Side Stick

    # Toms
    41: 'LT',  # Low Floor Tom
    43: 'LT',  # High Floor Tom
    45: 'MT',  # Low Tom
    47: 'MT',  # Low-Mid Tom
    48: 'HT',  # Hi-Mid Tom
    50: 'HT',  # High Tom

    # Hi-Hats
    42: 'CH',  # Closed Hi-Hat
    44: 'PH',  # Pedal Hi-Hat
    46: 'OH',  # Open Hi-Hat

    # Cymbals
    49: 'CC',  # Crash Cymbal 1
    57: 'CC',  # Crash Cymbal 2
    51: 'RC',  # Ride Cymbal 1
    59: 'RC',  # Ride Cymbal 2
    # Add any other mappings you need
}

# --- NEW: Create a reverse mapping from SMT name to MIDI number ---
SMT_TO_DRUM_MIDI = {v: k for k, v in DRUM_MIDI_TO_SMT.items()}


def get_duration_token(element):
    """Converts a music21 element's duration to an SMT duration token."""
    # Using fractions is robust for dotted notes, triplets, etc.
    duration_fraction = Fraction(element.duration.quarterLength)
    return f"[{duration_fraction.numerator}/{duration_fraction.denominator}]"

def convert_note_to_smt(element):
    """
    Converts a music21 element (Note, Rest, Chord) to its SMT string representation.
    This is now updated to handle Unpitched and PercussionChord objects.
    """
    if isinstance(element, music21.note.Rest):
        duration = element.duration.quarterLength
        return f"rest[{Fraction(duration).limit_denominator()}]"

    # --- UPDATED LOGIC FOR NOTES AND CHORDS ---
    elif isinstance(element, music21.note.Note): # Catches Note and Unpitched
        duration = element.duration.quarterLength
        pitch_name = 'unknown'
        
        if hasattr(element.pitch, 'midi'): # Old data format (standard Note)
            pitch_name = DRUM_MIDI_TO_SMT.get(element.pitch.midi, 'unknown')
        elif isinstance(element, music21.note.Unpitched): # New synthetic data (Unpitched)
            # Reconstruct the MIDI number from displayStep/displayOctave to find the SMT name
            p = music21.pitch.Pitch()
            p.step = element.displayStep
            p.octave = element.displayOctave
            # The generator script added 2 to the octave for display, so we subtract 2 to get the real MIDI value
            p.octave -= 2
            pitch_name = DRUM_MIDI_TO_SMT.get(p.midi, 'unknown')
            
        return f"note[{pitch_name},{Fraction(duration).limit_denominator()}]"

    elif isinstance(element, music21.chord.ChordBase): # Catches Chord and PercussionChord
        duration = element.duration.quarterLength
        
        # Get all MIDI numbers from the chord, whether it's a standard Chord or PercussionChord
        midi_numbers = []
        for n in element.notes:
            # --- THIS IS THE FIX ---
            # Check for the Unpitched type FIRST, since it doesn't have a .pitch attribute.
            if isinstance(n, music21.note.Unpitched): # Unpitched in a PercussionChord
                p = music21.pitch.Pitch()
                p.step = n.displayStep
                p.octave = n.displayOctave
                p.octave -= 2
                midi_numbers.append(p.midi)
            # Then, handle the case of a standard Note in a standard Chord.
            elif hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'): # Standard Note in a Chord
                midi_numbers.append(n.pitch.midi)

        # Convert MIDI numbers to SMT names and sort them alphabetically
        pitch_names = sorted([DRUM_MIDI_TO_SMT.get(m, 'unknown') for m in midi_numbers])
        
        if not pitch_names or 'unknown' in pitch_names:
            return None # Skip empty or unknown chords
            
        chord_content = "&".join(pitch_names)
        return f"note[{chord_content},{Fraction(duration).limit_denominator()}]"
        
    return None


def musicxml_to_smt(xml_path):
    """
    Parses a MusicXML file, finds the main drum part by its clef and name, 
    and returns a list of SMT strings for that part.
    """
    try:
        score = music21.converter.parse(xml_path)
    except Exception as e:
        print(f"Could not parse {xml_path}: {e}")
        return []

    # --- NEW LOGIC: Find the BEST drum part ---
    percussion_parts = []
    for part in score.parts:
        first_clef = part.flatten().getElementsByClass(music21.clef.Clef).first()
        if isinstance(first_clef, music21.clef.PercussionClef):
            percussion_parts.append(part)
    
    drum_part = None
    if not percussion_parts:
        return [] # No percussion parts found
    elif len(percussion_parts) == 1:
        drum_part = percussion_parts[0]
        print(f"  Found single drum part '{drum_part.partName}' in {xml_path.name}")
    else:
        # Multiple percussion parts found, try to find the main "drum set".
        print(f"  Found multiple percussion parts: {[p.partName for p in percussion_parts]}. Applying heuristic...")
        keywords = ['drum', 'kit', 'set', 'schlagzeug', 'batería', 'd. kit']
        for part in percussion_parts:
            if any(keyword in part.partName.lower() for keyword in keywords):
                drum_part = part
                print(f"  Selected '{drum_part.partName}' as the main drum part.")
                break
        
        if not drum_part:
            # Fallback: if no keyword match, just take the first one.
            drum_part = percussion_parts[0]
            print(f"  No keyword match. Defaulting to first part: '{drum_part.partName}'.")
    # --- END NEW LOGIC ---

    # --- DEFINITIVE PAGE ITERATION LOGIC ---
    page_smts = []
    current_page_events = []
    # We must iterate through the entire flattened part to find all elements in order.
    for element in drum_part.flatten():
        # A new page is explicitly marked by a PageLayout object.
        if isinstance(element, music21.layout.PageLayout) and element.isNew:
            # Finalize the previous page's SMT string.
            # This correctly handles tacet pages by appending an empty string.
            page_smts.append(" ".join(current_page_events))
            # Reset the list for the new page.
            current_page_events = []

        # Collect notes and rests.
        elif isinstance(element, (music21.note.GeneralNote, music21.note.Rest)):
            smt_token = convert_note_to_smt(element)
            if smt_token:
                current_page_events.append(smt_token)

    # After the loop, we must always append the content of the very last page.
    page_smts.append(" ".join(current_page_events))

    # A special case: if a score has no explicit page breaks at all, the loop
    # will finish and the list will have one giant SMT string. If we know there's
    # only one page of images, this is correct. But if there are multiple pages
    # of images, it means the MusicXML is poorly formatted and lacks page break info.
    # The mismatch check in main() will correctly catch this and skip the file.

    return page_smts


def pdf_to_images(pdf_path, output_dir):
    """Converts a PDF to a series of PNG images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, image in enumerate(images):
        path = os.path.join(output_dir, f"page_{i}.png")
        image.save(path, 'PNG')
        image_paths.append(path)
    return image_paths

def main():
    """
    Main function to build the dataset.
    """
    dataset = []
    with open(MANIFEST_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['do_or_mi'] == 'do':
                print(f"Processing drum-only file: {row['pdf']}")
                
                # --- THIS IS THE FIX ---
                # Use the pathlib '/' operator to ensure paths are Path objects
                pdf_path = PDF_DIR / row['pdf']
                xml_path = XML_DIR / row['musicxml']
                
                # 1. Convert PDF to images
                pdf_name_base = pdf_path.stem
                image_output_dir = OUTPUT_IMAGE_DIR / pdf_name_base
                image_paths = pdf_to_images(pdf_path, image_output_dir)
                
                # 2. Convert MusicXML to SMT strings (one per page)
                smt_strings_by_page = musicxml_to_smt(xml_path)
                
                # 3. Align images and SMT strings
                if len(image_paths) != len(smt_strings_by_page):
                    print(f"  [WARNING] Mismatch pages: {len(image_paths)} images vs {len(smt_strings_by_page)} SMT pages. Skipping.")
                    continue
                    
                for i, img_path in enumerate(image_paths):
                    dataset.append({
                        'image_path': img_path,
                        'smt_string': smt_strings_by_page[i]
                    })

    # 4. Save the final dataset manifest
    with open(FINAL_DATASET_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\nDataset creation complete. {len(dataset)} pairs created.")
    print(f"Manifest saved to {FINAL_DATASET_FILE}")


if __name__ == '__main__':
    main()