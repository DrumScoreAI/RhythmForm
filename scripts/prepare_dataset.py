import os
from pathlib import Path
import csv
import music21
import subprocess
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction

# --- Configuration ---
TRAINING_DATA_DIR = Path(os.environ.get('SFHOME', Path(__file__).parent.parent)) / 'training_data'
XML_DIR = TRAINING_DATA_DIR / 'musicxml'
OUTPUT_IMAGE_DIR = TRAINING_DATA_DIR / 'images'
MANIFEST_FILE = TRAINING_DATA_DIR / 'training_data.csv'
FINAL_DATASET_FILE = TRAINING_DATA_DIR / 'dataset.json'
MUSESCORE_PATH = os.environ.get("MUSESCORE_PATH", "mscore") # Use environment variable or default

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


# --- Symbolic Music Text (SMT) Generation ---

def get_instrument_name(unpitched_note):
    """Maps a music21 Unpitched note object back to our instrument name."""
    # --- UPDATED LOGIC ---
    # Use the reverse mapping SMT_TO_DRUM_MIDI to find the MIDI number,
    # then lookup the original SMT name from DRUM_MIDI_TO_SMT.
    midi_number = SMT_TO_DRUM_MIDI.get(unpitched_note.name, None)
    if midi_number is not None:
        return DRUM_MIDI_TO_SMT.get(midi_number, 'Unknown')
    
    return 'Unknown'

def measure_to_smt(measure, is_repeated=False):
    """
    Converts a single music21 measure to its SMT representation.
    If is_repeated is True, it returns a special repeat token.
    """
    if is_repeated:
        return "repeat[bar,1]"

    smt_tokens = []
    # Use .notesAndRests to iterate over all note, chord, and rest objects
    for element in measure.notesAndRests:
        token = convert_note_to_smt(element)
        if token:
            smt_tokens.append(token)
            
    return " ".join(smt_tokens)

def musicxml_to_smt(score_path, repeated_measures=None):
    """
    Converts a full MusicXML file to a single-line SMT string.
    Uses the list of repeated_measures to insert the correct token.
    """
    if repeated_measures is None:
        repeated_measures = []

    try:
        score = music21.converter.parse(score_path)
        drum_part = score.getElementById('drumset')
        if not drum_part:
            drum_part = score.parts[0]

        full_smt = []
        for measure in drum_part.getElementsByClass('Measure'):
            is_repeat = measure.number in repeated_measures
            smt = measure_to_smt(measure, is_repeated=is_repeat)
            full_smt.append(smt)
        
        return " measure_break ".join(full_smt)
    except Exception as e:
        print(f"Error parsing {score_path} with music21: {e}")
        return None

def create_repeat_modified_xml(original_xml_path, repeated_measures):
    """
    Injects <measure-style> tags for repeats into an XML file and returns the path to a temporary file.
    """
    if not repeated_measures:
        return original_xml_path

    try:
        # Register namespace to avoid ET adding 'ns0:' prefixes
        ET.register_namespace('', "http://www.musicxml.org/xsd/musicxml.xsd")
        tree = ET.parse(original_xml_path)
        root = tree.getroot()
        
        part = root.find('.//part[@id="drumset"]')
        if part is None:
            part = root.find('.//part') # Fallback
        if part is None:
            return original_xml_path # Should not happen

        measures = part.findall('measure')
        
        # Group consecutive repeated measures
        consecutive_repeats = []
        for m_num in sorted(repeated_measures):
            if not consecutive_repeats or m_num != consecutive_repeats[-1][-1] + 1:
                consecutive_repeats.append([m_num])
            else:
                consecutive_repeats[-1].append(m_num)

        for group in consecutive_repeats:
            start_measure_num = group[0]
            stop_measure_num = group[-1] + 1

            # Add 'start' tag to the first measure of the group
            start_measure_element = part.find(f".//measure[@number='{start_measure_num}']")
            if start_measure_element is not None:
                attributes = start_measure_element.find('attributes')
                if attributes is None:
                    attributes = ET.Element('attributes')
                    start_measure_element.insert(0, attributes)
                
                ms = ET.SubElement(attributes, 'measure-style')
                mr = ET.SubElement(ms, 'measure-repeat')
                mr.set('type', 'start')
                mr.text = '1'

            # Add 'stop' tag to the measure *after* the group
            stop_measure_element = part.find(f".//measure[@number='{stop_measure_num}']")
            if stop_measure_element is not None:
                attributes = stop_measure_element.find('attributes')
                if attributes is None:
                    attributes = ET.Element('attributes')
                    stop_measure_element.insert(0, attributes)

                ms = ET.SubElement(attributes, 'measure-style')
                mr = ET.SubElement(ms, 'measure-repeat')
                mr.set('type', 'stop')

        temp_xml_path = original_xml_path.with_suffix('.temp.xml')
        tree.write(temp_xml_path, encoding='UTF-8', xml_declaration=True)
        return temp_xml_path
    except Exception as e:
        print(f"  -> XML modification failed for {original_xml_path}: {e}")
        return original_xml_path


def process_file(xml_path):
    """
    Processes a single MusicXML file:
    1. Checks for a companion .json for repeat info.
    2. Generates SMT, using 'repeat[bar,1]' token if needed.
    3. Creates a temporary, modified XML if repeats exist.
    4. Renders the appropriate XML to PDF and then to PNG.
    5. Cleans up temporary files.
    Returns a dictionary for the dataset.json or None on failure.
    """
    print(f"Processing: {xml_path.name}")
    # Note: Temporary PDFs are created during rendering, but the final image is a PNG.
    pdf_path = OUTPUT_IMAGE_DIR / xml_path.with_suffix('.pdf').name
    png_path = OUTPUT_IMAGE_DIR / xml_path.with_suffix('.png').name

    # --- 1. Handle repeats and SMT generation ---
    json_path = xml_path.with_suffix('.json')
    repeated_measures = []
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                repeated_measures = data.get("repeated_measures", [])
                print(f"  -> Found repeat info: measures {repeated_measures}")
        except Exception as e:
            print(f"  -> Warning: Could not read JSON {json_path.name}: {e}")

    smt = musicxml_to_smt(xml_path, repeated_measures)
    if not smt:
        return None

    # --- 2. Render Image (with repeats if necessary) ---
    xml_to_render = xml_path
    temp_xml_path = None
    if repeated_measures:
        temp_xml_path = create_repeat_modified_xml(xml_path, repeated_measures)
        xml_to_render = temp_xml_path

    try:
        # Render the (potentially temporary) XML to PDF
        subprocess.run([
            MUSESCORE_PATH,
            '-o', str(pdf_path),
            str(xml_to_render)
        ], check=True, capture_output=True, text=True)

        # Convert PDF to PNG
        subprocess.run([
            'pdftoppm',
            '-png',
            '-singlefile',
            '-r', '150', # DPI
            str(pdf_path),
            str(png_path.with_suffix('')) # pdftoppm adds the suffix
        ], check=True, capture_output=True, text=True)

        if not png_path.exists():
             print(f"  -> Error: PNG not created for {xml_path.name}")
             return None

        print(f"  -> Successfully rendered to {png_path.name}")
        return {"image_path": str(png_path.relative_to(TRAINING_DATA_DIR)), "smt": smt}

    except subprocess.CalledProcessError as e:
        print(f"  -> Error during rendering of {xml_to_render.name}:")
        print(e.stderr)
        return None
    finally:
        # --- 3. Cleanup ---
        if pdf_path.exists():
            pdf_path.unlink()
        if temp_xml_path and temp_xml_path.exists():
            temp_xml_path.unlink()


def main():
    """
    Main function to generate the dataset using a manifest file.
    It reads training_data.csv, processes each entry in parallel to generate
    images and SMT, and creates a final dataset.json manifest.
    """
    # Ensure output directory exists
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_FILE.exists():
        print(f"Error: Manifest file not found at {MANIFEST_FILE}")
        return

    # Read the manifest to get the list of files to process
    xml_paths_to_process = []
    with open(MANIFEST_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # We are currently only processing "drums-only" files
            if row.get('do_or_mi') == 'do' and row.get('musicxml'):
                xml_path = XML_DIR / row['musicxml']
                if xml_path.exists():
                    xml_paths_to_process.append(xml_path)
                else:
                    print(f"  [WARNING] File '{row['musicxml']}' listed in manifest but not found in {XML_DIR}. Skipping.")
            elif row.get('do_or_mi') == 'do':
                print(f"  [WARNING] 'musicxml' field missing for drums-only entry in manifest. Skipping.")
            elif row.get('do_or_mi') == 'mi':
                print(f"  [INFO] Skipping 'mi' entry. Multi-instrument files are not yet supported.")

    print(f"Found {len(xml_paths_to_process)} 'drums-only' files to process from manifest.")

    dataset = []
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        future_to_xml = {executor.submit(process_file, xml_file): xml_file for xml_file in xml_paths_to_process}
        for future in as_completed(future_to_xml):
            result = future.result()
            if result:
                dataset.append(result)

    # Sort dataset by image path for consistency
    dataset.sort(key=lambda x: x['image_path'])

    # Save the final dataset manifest
    with open(FINAL_DATASET_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset creation complete. {len(dataset)} pairs created.")
    print(f"Manifest saved to {FINAL_DATASET_FILE}")


if __name__ == '__main__':
    main()