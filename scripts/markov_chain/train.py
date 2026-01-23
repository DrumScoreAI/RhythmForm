import argparse
import os
import pickle
from pathlib import Path
import json
from markov_chain import MarkovChain

def load_symbolic_texts(dataset_file):
    sequences = []
    with open(dataset_file, 'r') as f:
        json_data = json.load(f)
    for item in json_data:
        st_text = item.get('st', '').strip()
        if st_text:
            sequences.append(st_text)
    return sequences

def process_file(file_path, model):
    print(f"Processing {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split the content by spaces to get tokens, and handle multiple spaces
    tokens = [token for token in content.split(' ') if token]
    model.add_sequence(tokens)

def main():

    RHYTHMFORMHOME = os.environ.get('RHYTHMFORMHOME', Path(__file__).parent.parent)
    TRAINING_DATA_DIR = Path(RHYTHMFORMHOME) / 'training_data'
    FINE_TUNING_DIR = TRAINING_DATA_DIR / 'fine_tuning'
    DATASET_FILE = FINE_TUNING_DIR / 'finetune_dataset.json'

    parser = argparse.ArgumentParser(description="Train a Markov Chain on symbolic drum text.")
    parser.add_argument('--dataset', default=DATASET_FILE, help='JSON file with symbolic text data')
    parser.add_argument('--order', type=int, default=2, help='Order of the Markov chain')
    parser.add_argument('--output', default=TRAINING_DATA_DIR / 'markov_model.pkl', help='Output file for the trained model')
    args = parser.parse_args()

    sequences = load_symbolic_texts(args.dataset)
    print(f"Loaded {len(sequences)} sequences.")

    mc = MarkovChain(order=args.order)
    mc.train(sequences)
    print(f"Trained Markov Chain of order {args.order}.")

    with open(args.output, 'wb') as f:
        pickle.dump(mc, f)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()