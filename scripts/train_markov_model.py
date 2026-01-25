import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    from scripts.markov_chain.markov_chain import MarkovChain
except ImportError:
    print("Error: Could not import MarkovChain. Ensure scripts.markov_chain package exists.")
    sys.exit(1)

def train_model(input_path, output_path, order=2):
    print(f"Training Markov Chain (order={order}) on {input_path}...")
    
    # Initialize model
    model = MarkovChain(order=order)
    
    # Read corpus
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Preprocess: split lines into tokens
    # Assuming input is SMT format (space-separated tokens)
    sequences = [line.strip().split() for line in lines if line.strip()]
    
    print(f"Loaded {len(sequences)} sequences.")
    
    # Train
    model.train(sequences)
    print("Training complete.")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Markov Chain model on SMT data.")
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Path to the input corpus file (e.g., .smt file)"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='training_data/markov_model.pkl', 
        help="Path to save the pickled model"
    )
    parser.add_argument(
        '--order', 
        type=int, 
        default=2, 
        help="Order of the Markov Chain"
    )
    args = parser.parse_args()
    
    train_model(args.input, args.output, args.order)