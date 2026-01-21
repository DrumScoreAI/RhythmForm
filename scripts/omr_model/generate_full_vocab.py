import itertools
import json
import sys
from pathlib import Path
from fractions import Fraction

# Add project root to path to allow imports when run as a script
if __name__ == '__main__' and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    __package__ = "scripts.omr_model"

from . import config
from .utils import DRUM_DISPLAY_TO_SMT

# --- Possible note/rest durations (as Fractions to match utils.py) ---
DURATIONS = [Fraction(1, 4), Fraction(1, 2), Fraction(1, 1), Fraction(3, 2), Fraction(2, 1)]

# --- Possible time signatures ---
TIME_SIGNATURES = ['2/4', '3/4', '4/4', '5/4', '6/4', '3/8', '5/8', '6/8', '7/8', '9/8', '12/8']

# --- Metadata tokens ---
METADATA_TOKENS = [
    'clef[percussion]',
    'title[text]',
    'subtitle[text]',
    'composer[text]',
]

# Add time signatures to metadata tokens
METADATA_TOKENS.extend([f'time[{ts}]' for ts in TIME_SIGNATURES])

# --- Special tokens ---
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

def generate_note_tokens():
    tokens = set()
    
    # Get unique abbreviations from the mapping (e.g., 'BD', 'SD', 'HH')
    # DRUM_DISPLAY_TO_SMT maps (step, octave, notehead) -> Abbr
    abbrevs = sorted(list(set(DRUM_DISPLAY_TO_SMT.values())))

    # Single notes
    for abbr in abbrevs:
        for dur in DURATIONS:
            tokens.add(f"note[{abbr},{dur}]")
            
    # Chords (2 or 3 instruments)
    # We generate combinations of abbreviations
    for n in [2, 3]:
        for combo in itertools.combinations(abbrevs, n):
            # SMT format requires sorted, dot-separated abbreviations (e.g., 'BD.SD')
            inst_str = ".".join(sorted(combo))
            for dur in DURATIONS:
                tokens.add(f"note[{inst_str},{dur}]")
    return tokens

def generate_rest_tokens():
    return {f"rest[{dur}]" for dur in DURATIONS}

def main():
    tokens = set(SPECIAL_TOKENS)
    tokens.update(METADATA_TOKENS)
    note_tokens = generate_note_tokens()
    rest_tokens = generate_rest_tokens()
    tokens.update(note_tokens)
    tokens.update(rest_tokens)
    
    # Sort for consistency
    sorted_tokens = sorted(list(tokens))

    # Save to JSON
    out_path = config.TRAINING_DATA_DIR / 'full_tokenizer_vocab.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(sorted_tokens, f, indent=2)
    print(f"Wrote {len(tokens)} tokens to {out_path}")

    # Generate synthetic corpus for Markov training
    corpus_path = config.TRAINING_DATA_DIR / 'all_tokens_corpus.smt'
    with open(corpus_path, 'w') as f:
        # Combine notes and rests
        all_content_tokens = sorted(list(note_tokens.union(rest_tokens)))
        
        for ts in TIME_SIGNATURES:
            header = f"clef[percussion] time[{ts}] |"
            for token in all_content_tokens:
                f.write(f"{header} {token}\n")
        
        # Add a sequence with full metadata to ensure these tokens and their order are represented
        meta_header = "title[text] subtitle[text] composer[text] clef[percussion] time[4/4] |"
        f.write(f"{meta_header} rest[1]\n")
    print(f"Wrote synthetic corpus to {corpus_path}")

if __name__ == '__main__':
    main()
