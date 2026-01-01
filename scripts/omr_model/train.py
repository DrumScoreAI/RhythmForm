import torch
import torch.multiprocessing
import os
from . import config

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms # Import transforms
from tqdm import tqdm
import numpy as np
import argparse
import logging
import sys
from datetime import datetime

# Import all our custom modules
from . import config
from .dataset import ScoreDataset
from .tokenizer import StTokenizer
from .model import ImageToStModel

def setup_logging(log_file, log_to_stdout=False):
    """
    Sets up logging to file and optionally to stdout.
    """
    handlers = [logging.FileHandler(log_file)]
    if log_to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def collate_fn(batch, pad_token_id):
    """
    Custom collate function to pad sequences to the same length in a batch.
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Encode and pad the ST strings
    encoded_strings = [item['encoded_st'] for item in batch]
    max_len = max(len(s) for s in encoded_strings)
    
    padded_strings = []
    for s in encoded_strings:
        padded = s + [pad_token_id] * (max_len - len(s))
        padded_strings.append(torch.tensor(padded, dtype=torch.long))
        
    targets = torch.stack(padded_strings)
    
    return {'image': images, 'target': targets}

def parse_args():
    parser = argparse.ArgumentParser(description="Train the OMR model.")
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS, help='Number of data loader workers')
    parser.add_argument('--validation-split', type=float, default=config.VALIDATION_SPLIT, help='Validation data split ratio')
    parser.add_argument('--log-file', type=str, default=None, help='Path to log file')
    parser.add_argument('--log-stdout', action='store_true', help='Log to stdout as well')
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Logging Setup ---
    if args.log_file is None:
        # Default log file in the logs directory
        log_dir = config.TRAINING_DATA_DIR / 'logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = log_dir / f"train_log_{timestamp}.log"
    
    setup_logging(args.log_file, args.log_stdout)
    logging.info(f"Logging initialized. Writing to {args.log_file}")

    # Ensure checkpoint directory exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {config.CHECKPOINT_DIR}")

    """
    Main training function.
    """
    # --- 1. Setup ---
    logging.info(f"Using device: {config.DEVICE}")

    # Load tokenizer
    tokenizer = StTokenizer()
    tokenizer.load(config.TOKENIZER_VOCAB_PATH)
    pad_token_id = tokenizer.token_to_id['<pad>']
    
    # --- THIS IS THE FIX ---
    # Define a transform to convert PIL images to PyTorch Tensors.
    # ToTensor() converts a PIL Image (H x W x C) in the range [0, 255]
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    # We also need to resize the images to a fixed size to ensure they can be batched
    # and to fit in GPU memory.
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor()
    ])

    # Pass the tokenizer and the new transform to the dataset.
    # The dataset will apply ToTensor() to each image as it's loaded.
    full_dataset = ScoreDataset(
        manifest_path=config.DATASET_JSON_PATH,
        tokenizer=tokenizer,
        transform=transform
    )
    
    # --- Data Splitting ---
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.VALIDATION_SPLIT * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    # 3. Update the DataLoader to use a lambda function for the collate_fn
    logging.info(f"Creating DataLoaders with {args.num_workers} workers")
    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size, 
        sampler=validation_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # --- 2. Model, Loss, and Optimizer ---
    model = ImageToStModel(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize GradScaler for Mixed Precision Training
    scaler = torch.amp.GradScaler('cuda')

    # --- 3. Training Loop ---
    logging.info("--- Starting Training ---")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        # Tqdm writes to stderr by default. Changing to stdout.
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", file=sys.stdout)
        
        for batch in train_pbar:
            images = batch['image'].to(config.DEVICE)
            targets = batch['target'].to(config.DEVICE)
            
            # Decoder input is all but the last token (<eos>)
            # Ground truth is all but the first token (<sos>)
            decoder_input = targets[:, :-1]
            ground_truth = targets[:, 1:]
            
            # Forward pass with Mixed Precision
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = model(images, decoder_input)
                
                # Reshape for loss calculation
                # Output: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
                # Ground Truth: (batch, seq_len) -> (batch * seq_len)
                loss = criterion(output.reshape(-1, config.VOCAB_SIZE), ground_truth.reshape(-1))
            
            # Backward pass and optimization with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]", file=sys.stdout)
        
        with torch.no_grad():
            for batch in val_pbar:
                images = batch['image'].to(config.DEVICE)
                targets = batch['target'].to(config.DEVICE)
                
                decoder_input = targets[:, :-1]
                ground_truth = targets[:, 1:]
                
                output = model(images, decoder_input)
                loss = criterion(output.reshape(-1, config.VOCAB_SIZE), ground_truth.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # --- Save Checkpoints ---
        # 1. Save Latest (overwrite every epoch)
        last_path = config.CHECKPOINT_DIR / "model_last.pth"
        torch.save(model.state_dict(), last_path)
        
        # 2. Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = config.CHECKPOINT_DIR / "model_best.pth"
            torch.save(model.state_dict(), best_path)
            logging.info(f"New best model saved with val loss {best_val_loss:.4f}")

        # 3. Save Periodic (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            checkpoint_path = config.CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Periodic checkpoint saved: {checkpoint_path}")

    logging.info("--- Training Complete ---")


if __name__ == '__main__':
    main()