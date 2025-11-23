import os
from pathlib import Path
import torch
from multiprocessing import cpu_count

# --- Path Configuration ---
# Use the RHYTHMFORMHOME env var for the project root, with a fallback.
# This ensures all paths are resolved correctly from the project root.
PROJECT_ROOT = Path(os.environ.get('RHYTHMFORMHOME', Path(__file__).parent.parent.parent))
TRAINING_DATA_DIR = PROJECT_ROOT / 'training_data'
DATASET_JSON_PATH = TRAINING_DATA_DIR / 'dataset.json'
TOKENIZER_VOCAB_PATH = TRAINING_DATA_DIR / 'tokenizer_vocab.json'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

# --- Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # Adjust based on your GPU memory ( use 1 if running a CPU-based test )
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1

# --- THIS IS THE FIX ---
# On WSL, multiprocessing for the DataLoader can cause memory issues and crash the system.
# Setting NUM_WORKERS to 0 disables multiprocessing and makes the training stable,
# though it may be slightly slower.
NUM_WORKERS = cpu_count() // 2

# --- Model Hyperparameters ---
# These should match the model architecture defined in model.py
# Attempt to load vocab size from the tokenizer file
try:
    with open(TOKENIZER_VOCAB_PATH, 'r') as f:
        import json
        # --- THIS IS THE FIX ---
        # The tokenizer likely saves the vocabulary list directly, not in a {'vocab': ...} dict.
        # We load the JSON and assume it's the list of tokens.
        vocab_list = json.load(f)
        VOCAB_SIZE = len(vocab_list)
except FileNotFoundError:
    print(f"Warning: Tokenizer vocab not found at {TOKENIZER_VOCAB_PATH}. Using a default VOCAB_SIZE of 500.")
    print("Run 'python -m scripts.omr_model.tokenizer' to generate it.")
    VOCAB_SIZE = 500
except (json.JSONDecodeError, TypeError):
    # Handle cases where the file is not a simple list
    print(f"Warning: Could not determine vocab size from {TOKENIZER_VOCAB_PATH}. Using default.")
    VOCAB_SIZE = 500


D_MODEL = 256       # Embedding dimension
N_HEADS = 8         # Number of attention heads
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
