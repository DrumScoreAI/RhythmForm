import os
import json
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

if __name__ == '__main__' and __package__ is None:
    from os import path
    import sys
    # Add the project root to the python path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    # Now we can import as if we were a module
    from scripts.omr_model.dataset import ScoreDataset
    from scripts.omr_model import config
else:
    # Import the dataset class
    from .dataset import ScoreDataset
    from . import config

def _count_tokens_in_sample(sample):
    """Helper function to count tokens in a single sample for multiprocessing."""
    return Counter(sample['st_string'].strip().split(' '))

def _process_chunk(dataset_chunk):
    """Processes a chunk of the dataset and returns an aggregated token counter."""
    chunk_token_counts = Counter()
    for sample in dataset_chunk:
        chunk_token_counts.update(_count_tokens_in_sample(sample))
    return chunk_token_counts

class StTokenizer:
    """
    A tokenizer for converting ST strings to and from sequences of integer IDs.
    """
    def __init__(self):
        # Define the special tokens that the model requires.
        # The order here is important.
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = []
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, dataset, num_cores=1):
        """
        Builds or updates the vocabulary from a ScoreDataset object.
        If the tokenizer's vocab is already populated, it will only add new tokens.
        """
        # If the vocab is empty or only has special tokens, start it fresh.
        if len(self.vocab) <= len(self.special_tokens):
            self.vocab = self.special_tokens[:]
        
        # Count all tokens in the dataset in parallel
        token_counts = Counter()
        print(f"Counting tokens in dataset using {num_cores} cores...")

        # --- Parallelization Improvement ---
        # Split the dataset into chunks to be processed in parallel.
        # This is much more efficient than submitting a separate job for each sample.
        chunk_size = len(dataset) // num_cores
        if chunk_size == 0:
            chunk_size = 1 # Ensure at least one item per chunk
        print('here')
        
        chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

        print('now here')
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit each chunk for processing
            futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
            
            # Use tqdm for a progress bar over the futures
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing dataset chunks"):
                token_counts.update(future.result())
            
        # Add new tokens found in the dataset, ordered by frequency
        print("Building vocabulary from token counts...")
        for token, _ in token_counts.most_common():
            if token not in self.vocab:
                self.vocab.append(token)
        
        # Now, create the token-to-ID mapping from the final, ordered vocab list.
        print("Creating token-to-ID mapping...")
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        
        # Create the reverse mapping
        print("Creating ID-to-token mapping...")
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, st_string):
        """Converts an ST string to a list of integer IDs."""
        tokens = st_string.strip().split(' ')
        
        # Ensure that any token not found in the vocabulary is mapped
        # to the ID of the '<unk>' token.
        unk_id = self.token_to_id['<unk>']
        ids = [self.token_to_id.get(token, unk_id) for token in tokens]

        # Add Start Of Sequence and End Of Sequence tokens
        # We prepend <sos> and append <eos>
        final_ids = [self.token_to_id['<sos>']] + ids + [self.token_to_id['<eos>']]
        
        return final_ids

    def decode(self, ids):
        """
        Decodes a list of integer IDs back into an ST string.
        Stops at the first EOS token.
        """
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, '<unk>')
            if token == '<eos>':
                break
            if token not in ['<sos>', '<pad>']:
                tokens.append(token)
        return " ".join(tokens)

    def save(self, filepath):
        """Saves the tokenizer's ordered vocabulary list to a JSON file with enhanced debugging."""
        try:
            # --- Systematic Debugging: Step 1: Resolve and print the absolute path ---
            abs_path = Path(filepath).resolve()
            print(f"--> Attempting to save tokenizer to absolute path: {abs_path}")

            # --- Systematic Debugging: Step 2: Check if parent directory exists ---
            parent_dir = abs_path.parent
            if not parent_dir.exists():
                print(f"--> ERROR: Parent directory does not exist: {parent_dir}")
                return # Stop if the directory isn't there

            # --- Systematic Debugging: Step 3: Attempt to write the file ---
            print(f"--> Writing {len(self.vocab)} tokens to the vocabulary file...")
            with open(abs_path, 'w') as f:
                json.dump(self.vocab, f, indent=2)
            
            print(f"--> SUCCESS: Tokenizer vocabulary saved to {abs_path}")

        except (IOError, OSError) as e:
            # --- Systematic Debugging: Step 4: Catch and report any errors ---
            print(f"--> ERROR: Failed to save tokenizer file. An I/O error occurred: {e}")
        except Exception as e:
            print(f"--> ERROR: An unexpected error occurred during save: {e}")

    def load(self, filepath):
        """Loads the tokenizer's vocabulary from a JSON file."""
        # --- THIS IS THE FIX (Part 3) ---
        # Load the ordered vocabulary list.
        try:
            with open(filepath, 'r') as f:
                self.vocab = json.load(f)
            
            # Rebuild the mappings from the loaded vocabulary list.
            self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
            print(f"Tokenizer vocabulary loaded from {filepath}")
        except FileNotFoundError:
            print(f"Tokenizer vocabulary not found at {filepath}. Initializing an empty tokenizer.")
            # Initialize with special tokens only, so it's in a consistent state.
            self.vocab = self.special_tokens[:]
            self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}


# This block allows you to test the tokenizer by running `python omr_model/tokenizer.py`
if __name__ == '__main__':
    import multiprocessing
    # Set the start method to 'fork' to avoid potential issues
    # with how child processes are created, especially when a script is
    # called from another script.
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        # This will happen if the start method has already been set.
        # It's safe to ignore in that case.
        pass

    parser = argparse.ArgumentParser(description="Build and test the tokenizer.")
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to use for parallel processing (default: 1)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force the vocabulary to be rebuilt even if it already exists."
    )
    args = parser.parse_args()

    # Define paths
    TOKENIZER_SAVE_PATH = config.TOKENIZER_VOCAB_PATH
    DATASET_JSON_PATH = config.DATASET_JSON_PATH

    # Handle --force-rebuild by deleting the old vocab file
    if args.force_rebuild and os.path.exists(TOKENIZER_SAVE_PATH):
        print("--- Force rebuild requested. Deleting existing vocabulary. ---")
        os.remove(TOKENIZER_SAVE_PATH)

    # 1. Load tokenizer. This will initialize it if it doesn't exist.
    print("--- Loading tokenizer ---")
    tokenizer = StTokenizer()
    tokenizer.load(TOKENIZER_SAVE_PATH)
    original_vocab_size = tokenizer.vocab_size

    # 2. Build/update vocabulary from the full dataset
    print("\n--- Building/updating vocabulary from dataset ---")
    if not os.path.exists(DATASET_JSON_PATH):
        raise FileNotFoundError(
            f"Dataset manifest not found at {DATASET_JSON_PATH}. "
            "Please run the data preparation script first."
        )
    score_dataset = ScoreDataset(manifest_path=DATASET_JSON_PATH)
    tokenizer.build_vocab(score_dataset, num_cores=args.cores)

    # 3. Save the potentially updated tokenizer
    print("\n--- Saving tokenizer ---")
    tokenizer.save(TOKENIZER_SAVE_PATH)
    print(f"Vocabulary updated. Size changed from {original_vocab_size} to {tokenizer.vocab_size}.")

    # 4. Verification and testing
    print(f"\nFinal vocabulary size: {tokenizer.vocab_size}")
    print(f"First 10 tokens: {tokenizer.vocab[:10]}")
    
    if len(score_dataset) > 0:
        print("\n--- Testing encoding and decoding ---")
        sample_string = score_dataset[0]['st_string']
        print(f"Original string (first 80 chars): {sample_string[:80]}...")
        
        encoded_ids = tokenizer.encode(sample_string)
        print(f"Encoded IDs (first 20): {encoded_ids[:20]}...")
        
        decoded_string = tokenizer.decode(encoded_ids)
        print(f"Decoded string (first 80 chars): {decoded_string[:80]}...")
        
        assert sample_string == decoded_string
        print("✅ Encode/Decode test passed!")
    else:
        print("Dataset is empty. Skipping encode/decode test.")

    print("\n--- Verifying tokenizer integrity ---")
    new_tokenizer = StTokenizer()
    new_tokenizer.load(TOKENIZER_SAVE_PATH)
    print(f"Loaded vocabulary size: {new_tokenizer.vocab_size}")
    assert tokenizer.vocab_size == new_tokenizer.vocab_size
    assert tokenizer.vocab == new_tokenizer.vocab
    print("✅ Tokenizer integrity test passed!")