import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path to allow imports when run as a script
if __name__ == '__main__' and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scripts.omr_model import config

# Attempt to import the MarkovChain class
try:
    from scripts.markov_chain import MarkovChain
except ImportError:
    # Fallback for different directory structures or if running from root
    try:
        from scripts.markov_chain.markov_chain import MarkovChain
    except ImportError:
        print("Error: Could not import MarkovChain. Please ensure 'scripts.markov_chain' is accessible.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train a Markov Chain model on the SMT corpus.")
    parser.add_argument('--corpus', type=Path, default=config.TRAINING_DATA_DIR / 'markov_training_corpus.smt',
                        help="Path to the training corpus file.")
    parser.add_argument('--output', type=Path, default=config.TRAINING_DATA_DIR / 'markov_model.pkl',
                        help="Path to save the trained model pickle.")
    parser.add_argument('--order', type=int, default=1, 
                        help="Order of the Markov Chain (default: 1).")
    
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: Corpus file not found at {args.corpus}")
        print("Please run 'scripts/build_markov_corpus.py' first.")
        return

    print(f"Reading corpus from {args.corpus}...")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        # Read lines and split into tokens
        sequences = [line.strip().split() for line in f if line.strip()]

    print(f"Loaded {len(sequences)} sequences.")
    print(f"Training Markov Chain (Order {args.order})...")
    
    model = MarkovChain(order=args.order)
    model.train(sequences)

    # Save the model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Successfully trained and saved Markov model to {args.output}")

if __name__ == '__main__':
    main()