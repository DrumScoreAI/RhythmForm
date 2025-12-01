import argparse
import pickle
from .markov_chain import MarkovChain

def generate_sequence(model, length=100, start_token=None):
    return model.generate(length=length, start_token=start_token)

def generate_musicxml(sequence):
    # Placeholder function: Convert the generated sequence into MusicXML format.
    # The actual implementation would depend on the specific symbolic text format used.
    musicxml_content = f"<musicxml>\n<!-- Generated Sequence -->\n{sequence}\n</musicxml>"
    return musicxml_content

def main():
    parser = argparse.ArgumentParser(description="Generate a sequence using a trained MarkovChain model.")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained MarkovChain model (pickle file)')
    parser.add_argument('--length', type=int, default=100, help='Length of the sequence to generate')
    parser.add_argument('--start', type=str, default=None, help='Optional start token')
    parser.add_argument('--output', type=str, default=None, help='Output file to save the generated sequence (prints to stdout if not set)')
    args = parser.parse_args()

    # Load the trained MarkovChain model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    if not isinstance(model, MarkovChain):
        raise ValueError('Loaded object is not a MarkovChain instance')

    # Generate sequence
    sequence = generate_sequence(model, length=args.length, start_token=args.start)

    # Output
    if args.output:
        with open(args.output, 'w') as out_f:
            for item in sequence:
                out_f.write(f"{item}")
            out_f.write('\n')
    else:
        for item in sequence:
            print(item, end='')
        print()

if __name__ == "__main__":
    main()
