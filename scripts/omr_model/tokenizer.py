import os
import json
from pathlib import Path
from collections import Counter

# Import the dataset class
from .dataset import ScoreDataset
from . import config

class StTokenizer:
    """
    A tokenizer for converting ST strings to and from sequences of integer IDs.
    """
    def __init__(self):
        # --- THIS IS THE FIX (Part 1) ---
        # Define the special tokens that the model requires.
        # The order here is important.
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = []
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, dataset):
        """
        Builds the vocabulary from a ScoreDataset object.
        """
        # --- THIS IS THE FIX (Part 2) ---
        # Start the vocabulary with the essential special tokens.
        self.vocab = self.special_tokens[:]
        
        # Count all tokens in the dataset
        token_counts = Counter()
        print("Counting tokens in dataset...")
        for i, sample in enumerate(dataset):
            print(f"  Processing sample {i+1}/{len(dataset)}", end='\r')
            tokens = sample['st_string'].strip().split(' ')
            token_counts.update(tokens)
        print("")
            
        # Add new tokens found in the dataset, ordered by frequency
        print("Building vocabulary...")
        i=0
        for token, _ in token_counts.most_common():
            i += 1
            print(f"  Adding token {i}: {token}", end='\r')
            if token not in self.vocab:
                self.vocab.append(token)
        print("")
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
        
        # --- THIS IS THE FIX ---
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
        """Saves the tokenizer's ordered vocabulary list to a JSON file."""
        # --- THIS IS THE FIX (Part 2) ---
        # Save the ordered vocabulary list, not the dictionary.
        # This is the single source of truth for vocab size and token order.
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        print(f"Tokenizer vocabulary saved to {filepath}")

    def load(self, filepath):
        """Loads the tokenizer's vocabulary from a JSON file."""
        # --- THIS IS THE FIX (Part 3) ---
        # Load the ordered vocabulary list.
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        
        # Rebuild the mappings from the loaded vocabulary list.
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        print(f"Tokenizer vocabulary loaded from {filepath}")


# This block allows you to test the tokenizer by running `python omr_model/tokenizer.py`
if __name__ == '__main__':
    # Define where to save the tokenizer vocab
    PROJECT_ROOT = config.PROJECT_ROOT
    TOKENIZER_SAVE_PATH = config.TOKENIZER_VOCAB_PATH
    DATASET_JSON_PATH = config.DATASET_JSON_PATH

    # 1. Build vocabulary from scratch
    print("--- Building tokenizer from dataset ---")
    score_dataset = ScoreDataset(manifest_path=DATASET_JSON_PATH)
    tokenizer = StTokenizer()
    tokenizer.build_vocab(score_dataset)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"First 10 tokens: {tokenizer.vocab[:10]}")
    
    # 2. Test encoding and decoding
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

    # 3. Save the tokenizer
    print("\n--- Saving tokenizer ---")
    tokenizer.save(TOKENIZER_SAVE_PATH)

    # 4. Load the tokenizer and verify it works
    print("\n--- Loading tokenizer ---")
    new_tokenizer = StTokenizer()
    new_tokenizer.load(TOKENIZER_SAVE_PATH)
    print(f"Loaded vocabulary size: {new_tokenizer.vocab_size}")
    assert tokenizer.vocab_size == new_tokenizer.vocab_size
    print("✅ Load test passed!")