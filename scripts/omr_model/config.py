import os
import json
from pathlib import Path
import torch

# --- Project Paths ---
# Assumes SFHOME is set, otherwise defaults to the project's parent directory
PROJECT_ROOT = Path(os.environ.get('SFHOME', Path(__file__).parent.parent))
DATA_DIR = PROJECT_ROOT / 'training_data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

# Ensure checkpoint directory exists
CHECKPOINT_DIR.mkdir(exist_ok=True)

# --- Data Files ---
DATASET_JSON_PATH = DATA_DIR / 'dataset.json'
TOKENIZER_VOCAB_PATH = DATA_DIR / 'tokenizer_vocab.json'

# --- Data Parameters ---
IMG_HEIGHT = 1024
IMG_WIDTH = 512

# Load tokenizer to get vocab size dynamically
try:
    with open(TOKENIZER_VOCAB_PATH, 'r') as f:
        vocab_data = json.load(f)
    VOCAB_SIZE = len(vocab_data['token_to_id'])
except FileNotFoundError:
    print(f"Warning: Tokenizer vocab not found at {TOKENIZER_VOCAB_PATH}. Using a default VOCAB_SIZE of 500.")
    print("Run 'python -m omr_model.tokenizer' to generate it.")
    VOCAB_SIZE = 500

# --- Model Hyperparameters ---
# These are small values for a quick test run on a small dataset.
# They should be increased for a full training run.
D_MODEL = 256             # The dimension of the transformer model
NHEAD = 4                 # The number of attention heads
NUM_ENCODER_LAYERS = 3    # The number of encoder layers
NUM_DECODER_LAYERS = 3    # The number of decoder layers
DIM_FEEDFORWARD = 1024    # The dimension of the feedforward network
DROPOUT = 0.1
PATCH_SIZE = 32           # The size of the image patches

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4            # Number of samples per batch (adjust based on your GPU memory)
NUM_EPOCHS = 100          # Number of epochs for the overfitting test
LEARNING_RATE = 1e-4      # Learning rate for the optimizer