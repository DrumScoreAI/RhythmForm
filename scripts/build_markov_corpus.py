import argparse
from pathlib import Path
import sys

# Add project root to path to allow imports when run as a script
if __name__ == '__main__' and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scripts.omr_model import config
from scripts.omr_model.utils import normalize_smt_for_markov

def main():
    parser = argparse.ArgumentParser(description="Combine and normalize SMT files for Markov training.")
    parser.add_argument('--input-dir', type=Path, default=config.TRAINING_DATA_DIR / 'smt', 
                        help="Directory containing real SMT files.")
    parser.add_argument('--output', type=Path, default=config.TRAINING_DATA_DIR / 'markov_training_corpus.smt', 
                        help="Path to the output corpus file.")
    parser.add_argument('--vocab-corpus', type=Path, default=config.TRAINING_DATA_DIR / 'all_tokens_corpus.smt',
                        help="Path to the synthetic vocab corpus (from generate_full_vocab.py).")
    
    args = parser.parse_args()
    
    lines = []
    
    # 1. Read and normalize real data
    if args.input_dir.exists():
        smt_files = list(args.input_dir.glob("*.smt"))
        print(f"Found {len(smt_files)} SMT files in {args.input_dir}")
        
        for smt_file in smt_files:
            try:
                with open(smt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        normalized = normalize_smt_for_markov(content)
                        lines.append(normalized)
            except Exception as e:
                print(f"Error reading {smt_file}: {e}")
    else:
        print(f"Warning: Input directory {args.input_dir} does not exist.")

    # 2. Append synthetic vocab corpus
    if args.vocab_corpus.exists():
        print(f"Appending synthetic vocab corpus from {args.vocab_corpus}")
        with open(args.vocab_corpus, 'r', encoding='utf-8') as f:
            synthetic_lines = f.read().splitlines()
            lines.extend(synthetic_lines)
    else:
        print(f"Warning: Synthetic vocab corpus not found at {args.vocab_corpus}")

    # 3. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    print(f"Successfully wrote {len(lines)} lines to {args.output}")

if __name__ == '__main__':
    main()