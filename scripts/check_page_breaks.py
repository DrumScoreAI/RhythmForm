import os
import music21
import argparse
from pathlib import Path

def analyze_page_break(xml_path):
    """
    Parses a MusicXML file and finds the first musical event on page 2.
    """
    try:
        score = music21.converter.parse(xml_path)
    except Exception as e:
        print(f"Could not parse {xml_path.name}: {e}")
        return

    # We need to find the drum part first, using the same logic as prepare_dataset.py
    percussion_parts = []
    for part in score.parts:
        first_clef = part.flatten().getElementsByClass(music21.clef.Clef).first()
        if isinstance(first_clef, music21.clef.PercussionClef):
            percussion_parts.append(part)
    
    if not percussion_parts:
        return # Skip files with no drums

    # Use the same heuristic to select the main drum part
    drum_part = percussion_parts[0]
    if len(percussion_parts) > 1:
        keywords = ['drum', 'kit', 'set', 'schlagzeug']
        for part in percussion_parts:
            part_name = part.partName.lower() if part.partName else ""
            if any(keyword in part_name for keyword in keywords):
                drum_part = part
                break

    # --- Find the first event on page 2 ---
    page_number = 1
    found_event_on_page_2 = False

    for element in drum_part.flatten():
        # Check for a page break
        if isinstance(element, music21.layout.PageLayout) and element.isNew:
            page_number += 1
            # If we've just entered page 2, we start looking for the first note
            if page_number == 2:
                continue # Continue to the next element which might be the note

        # If we are on page 2 and find a note/rest, this is our target
        if page_number == 2 and isinstance(element, (music21.note.GeneralNote, music21.note.Rest)):
            measure_number = element.measureNumber if element.measureNumber is not None else "N/A"
            
            description = f"Measure {measure_number}: "
            if isinstance(element, music21.note.Rest):
                description += f"A rest with duration {element.duration.quarterLength}"
            else:
                description += f"A note/chord with duration {element.duration.quarterLength}"

            print(f"File: {xml_path.name}")
            print(f"  - First event on Page 2: {description}\n")
            
            found_event_on_page_2 = True
            break # We found what we need, so we can stop processing this file

    if not found_event_on_page_2 and page_number > 1:
        print(f"File: {xml_path.name}")
        print(f"  - Page 2 appears to be tacet (no notes found).\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze MusicXML files to find the first event on page 2.")
    parser.add_argument('xml_directory', type=str, help='The directory containing your MusicXML files.')
    args = parser.parse_args()

    xml_dir = Path(args.xml_directory)
    if not xml_dir.is_dir():
        print(f"Error: Directory not found at {xml_dir}")
        return

    print(f"--- Analyzing Page Breaks in {xml_dir} ---\n")
    for filename in sorted(os.listdir(xml_dir)):
        if filename.lower().endswith(('.xml', '.musicxml')):
            analyze_page_break(xml_dir / filename)

if __name__ == '__main__':
    main()