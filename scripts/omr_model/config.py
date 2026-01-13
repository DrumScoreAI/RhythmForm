import os
from pathlib import Path
import torch
from multiprocessing import cpu_count
from glob import glob
import subprocess

# --- Path Configuration ---
# Use the RHYTHMFORMHOME env var for the project root, with a fallback.
# This ensures all paths are resolved correctly from the project root.
PROJECT_ROOT = Path(os.environ.get('RHYTHMFORMHOME', Path(__file__).parent.parent.parent))
TRAINING_DATA_DIR = PROJECT_ROOT / 'training_data'
FINETUNING_DIR = TRAINING_DATA_DIR / 'fine_tuning'
fine_tuning = True
if fine_tuning:
    TRAINING_DATA_DIR = FINETUNING_DIR
    DATASET_JSON_PATH = TRAINING_DATA_DIR / 'finetune_dataset.json'
    MANIFEST_FILE = ''
    TOKENIZER_VOCAB_PATH = TRAINING_DATA_DIR / 'finetune_tokenizer_vocab.json'
    CHECKPOINT_DIR = TRAINING_DATA_DIR / 'checkpoints' / 'fine_tuned'
else:
    TOKENIZER_VOCAB_PATH = TRAINING_DATA_DIR / 'tokenizer_vocab.json'
    MANIFEST_FILE = TRAINING_DATA_DIR / 'training_data.csv'
    CHECKPOINT_DIR = TRAINING_DATA_DIR / 'checkpoints'
XML_DIR = TRAINING_DATA_DIR / 'musicxml'
DATA_IMAGES_DIR = TRAINING_DATA_DIR / 'images'
PDF_OUTPUT_DIR = TRAINING_DATA_DIR / 'pdfs'


pretrained_models = glob(str(CHECKPOINT_DIR / 'model_epoch_*.pth'))
if pretrained_models:
    for model_path in pretrained_models:
        highest_epoch = max(int(Path(p).stem.split('_')[-1]) for p in pretrained_models)
    FINETUNE_PRETRAINED_MODEL_PATH = CHECKPOINT_DIR / f'finetuned_model_epoch_{highest_epoch}.pth'
else:
    FINETUNE_PRETRAINED_MODEL_PATH = CHECKPOINT_DIR / '..' / 'model_best.pth'

# --- Training Configuration ---
# Allow forcing CPU for inference/testing
FORCE_CPU = os.environ.get("FORCE_CPU", "false").lower() == "true"
DEVICE = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1 # Adjust based on your GPU memory ( use 1 if running a CPU-based test )
NUM_EPOCHS = 50
FINE_NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
FINE_LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.1

IMAGE_DPI = 100  # DPI for image rendering
IMG_HEIGHT = 256  # Height to resize images
IMG_WIDTH = 1024  # Width to resize images

# --- THIS IS THE FIX ---
# On WSL, multiprocessing for the DataLoader can cause memory issues and crash the system.
# We will reduce the number of workers in that environment.
if subprocess.getoutput('uname -r').lower().find('microsoft') != -1:
    NUM_WORKERS = cpu_count() // 2
else:
    NUM_WORKERS = 32
    

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
    # print(f"Warning: Tokenizer vocab not found at {TOKENIZER_VOCAB_PATH}. Using a default VOCAB_SIZE of 500.")
    # print("Run 'python -m scripts.omr_model.tokenizer' to generate it.")
    VOCAB_SIZE = 500
except (json.JSONDecodeError, TypeError):
    # Handle cases where the file is not a simple list
    # print(f"Warning: Could not determine vocab size from {TOKENIZER_VOCAB_PATH}. Using default.")
    VOCAB_SIZE = 500


D_MODEL = 256       # Embedding dimension
NHEAD = 8         # Number of attention heads
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5