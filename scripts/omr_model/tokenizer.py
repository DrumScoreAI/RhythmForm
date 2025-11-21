import os
import json
from pathlib import Path
from collections import Counter

# Import the dataset class we just created
from .dataset import ScoreDataset, DATASET_JSON_PATH

class SmtTokenizer:
    """
    A tokenizer for converting SMT strings to and from sequences of integer IDs.
    """
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab = []
        
        # Define special tokens
        self.special_tokens = {
            '<pad>': 0,  # Padding token
            '<sos>': 1,  # Start of sequence
            '<eos>': 2,  # End of sequence
            '<unk>': 3,  # Unknown token
        }

    def build_vocab(self, dataset):
        """
        Builds the vocabulary from a ScoreDataset object.
        
        Args:
            dataset (ScoreDataset): The dataset containing the smt_strings.
        """
        # Start with special tokens
        self.token_to_id = self.special_tokens.copy()
        self.vocab = list(self.special_tokens.keys())
        
        # Count all tokens in the dataset
        token_counts = Counter()
        for sample in dataset:
            tokens = sample['smt_string'].split()
            token_counts.update(tokens)
            
        # Add tokens to the vocabulary, sorted by frequency for good measure
        for token, _ in token_counts.most_common():
            if token not in self.token_to_id:
                self.vocab.append(token)
                self.token_to_id[token] = len(self.vocab) - 1
        
        # Create the reverse mapping
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, smt_string):
        """
        Encodes an SMT string into a list of integer IDs, adding SOS and EOS tokens.
        """
        tokens = smt_string.split()
        encoded = [self.token_to_id['<sos>']]
        for token in tokens:
            encoded.append(self.token_to_id.get(token, self.token_to_id['<unk>']))
        encoded.append(self.token_to_id['<eos>'])
        return encoded

    def decode(self, ids):
        """
        Decodes a list of integer IDs back into an SMT string.
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
        """Saves the tokenizer's vocabulary to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump({'token_to_id': self.token_to_id}, f, indent=2)
        print(f"Tokenizer vocabulary saved to {filepath}")

    def load(self, filepath):
        """Loads the tokenizer's vocabulary from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.token_to_id = data['token_to_id']
        self.vocab = list(self.token_to_id.keys())
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        print(f"Tokenizer vocabulary loaded from {filepath}")


# This block allows you to test the tokenizer by running `python omr_model/tokenizer.py`
if __name__ == '__main__':
    # Define where to save the tokenizer vocab
    PROJECT_ROOT = Path(os.environ.get('SFHOME', Path(__file__).parent.parent))
    TOKENIZER_SAVE_PATH = PROJECT_ROOT / 'training_data' / 'tokenizer_vocab.json'

    # 1. Build vocabulary from scratch
    print("--- Building tokenizer from dataset ---")
    score_dataset = ScoreDataset(manifest_path=DATASET_JSON_PATH)
    tokenizer = SmtTokenizer()
    tokenizer.build_vocab(score_dataset)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"First 10 tokens: {tokenizer.vocab[:10]}")
    
    # 2. Test encoding and decoding
    if len(score_dataset) > 0:
        print("\n--- Testing encoding and decoding ---")
        sample_string = score_dataset[0]['smt_string']
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
    new_tokenizer = SmtTokenizer()
    new_tokenizer.load(TOKENIZER_SAVE_PATH)
    print(f"Loaded vocabulary size: {new_tokenizer.vocab_size}")
    assert tokenizer.vocab_size == new_tokenizer.vocab_size
    print("✅ Load test passed!")