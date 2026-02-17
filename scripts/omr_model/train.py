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
from functools import partial

# Import all our custom modules
from . import config
from .dataset import ScoreDataset
from .tokenizer import SmtTokenizer
from .model import ImageToSmtModel
from .predict import beam_search_predict

# Custom Transform for adding Gaussian Noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, p={self.p})'


def setup_logging(log_file, log_to_stdout=False):
    """
    Sets up logging to file and optionally to stdout.
    """
    handlers = [logging.FileHandler(log_file)]
    if log_to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

def collate_fn(batch, pad_token_id):
    """
    Custom collate function to pad sequences to the same length in a batch.
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Encode and pad the SMT strings
    encoded_strings = [item['encoded_smt'] for item in batch]
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
    parser.add_argument('--resume-from', type=str, default=None, help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=1.0, help='Initial teacher forcing ratio.')
    parser.add_argument('--teacher-forcing-decay', type=float, default=0.95, help='Teacher forcing decay factor per epoch.')
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

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logging.info(f"Random seed set to {args.seed} for reproducibility.")

    # Ensure checkpoint directory exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {config.CHECKPOINT_DIR}")

    """
    Main training function.
    """
    # --- 1. Setup ---
    logging.info(f"Using device: {config.DEVICE}")

    # Load tokenizer
    tokenizer = SmtTokenizer()
    tokenizer.load(config.TOKENIZER_VOCAB_PATH)
    pad_token_id = tokenizer.token_to_id['<pad>']
    
    # --- THIS IS THE FIX ---
    # Define a transform to convert PIL images to PyTorch Tensors.
    # ToTensor() converts a PIL Image (H x W x C) in the range [0, 255]
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    # We also need to resize the images to a fixed size to ensure they can be batched
    # and to fit in GPU memory.

    # Define the four augmentations
    aug_transforms = [
        ("RandomAffine", transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5)),
        ("ElasticTransform", transforms.RandomApply([transforms.ElasticTransform(alpha=75.0, sigma=5.0)], p=0.5)),
        ("RandomPerspective", transforms.RandomPerspective(distortion_scale=0.2, p=0.5)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.4, contrast=0.4)),
    ]

    # This will be set per epoch
    active_aug_names = []
    active_aug_transforms = []

    def build_train_transform():
        # Randomly select 2 augmentations for this epoch
        import random
        selected = random.sample(aug_transforms, 2)
        nonlocal active_aug_names, active_aug_transforms
        active_aug_names = [name for name, _ in selected]
        active_aug_transforms = [t for _, t in selected]
        # Compose the transform pipeline
        return transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            *active_aug_transforms,
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.05, p=0.7)
        ])

    train_transform = build_train_transform()

    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor()
    ])

    # Create two separate dataset instances for training and validation
    # each with its own transform.
    train_dataset = ScoreDataset(
        manifest_path=config.DATASET_JSON_PATH,
        tokenizer=tokenizer,
        transform=train_transform
    )
    val_dataset = ScoreDataset(
        manifest_path=config.DATASET_JSON_PATH,
        tokenizer=tokenizer,
        transform=val_transform
    )
    
    # --- Data Splitting ---
    dataset_size = len(train_dataset) # Both datasets have the same size initially
    indices = list(range(dataset_size))
    split = int(np.floor(config.VALIDATION_SPLIT * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create samplers that will draw indices from the correct splits
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    # --- Grab a single validation image for qualitative prediction ---
    fixed_val_image = None
    ground_truth_smt = None
    if val_indices:
        # Get the dictionary for the first validation sample
        val_sample = val_dataset[val_indices[0]] # Use the val_dataset with non-augmenting transform
        # Extract the image tensor and move it to the correct device
        fixed_val_image = val_sample['image'].to(config.DEVICE)
        # Decode the ground truth SMT string for comparison
        ground_truth_smt = tokenizer.decode(val_sample['encoded_smt'])
        logging.info(f"Using validation image index {val_indices[0]} for qualitative prediction checks during training.")


    # 3. Update the DataLoader to use the correct dataset and sampler
    logging.info(f"Creating DataLoaders with {args.num_workers} workers")

    collate_with_pad = partial(collate_fn, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset, # Use the training dataset
        batch_size=args.batch_size,
        sampler=train_sampler, # The sampler will select from the correct indices
        num_workers=args.num_workers,
        collate_fn=collate_with_pad,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, # Use the validation dataset
        batch_size=args.batch_size, 
        sampler=validation_sampler, # The sampler will select from the correct indices
        num_workers=args.num_workers,
        collate_fn=collate_with_pad,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # --- 2. Model, Loss, and Optimizer ---
    model = ImageToSmtModel(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    # --- DataParallel for Multi-GPU ---
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPU(s)")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=config.WEIGHT_DECAY)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Initialize GradScaler for Mixed Precision Training
    scaler = torch.amp.GradScaler('cuda')

    # --- Checkpointing Setup ---
    start_epoch = 0
    best_val_loss = float('inf')
    best_token_accuracy = -1.0
    
    # --- Resume from Checkpoint ---
    if args.resume_from:
        if os.path.exists(args.resume_from):
            logging.info(f"Resuming training from checkpoint: {args.resume_from}")
            # Note: We are not using weights_only=True as we need to load the states of the 
            # optimizer, scheduler, and scaler. Only load checkpoints from trusted sources.
            checkpoint = torch.load(args.resume_from, map_location=config.DEVICE, weights_only=False)
            
            model_state_dict = checkpoint['model_state_dict']

            # Handle DataParallel prefix
            is_dataparallel = isinstance(model, nn.DataParallel)
            checkpoint_is_dataparallel = all(k.startswith('module.') for k in model_state_dict.keys())

            if is_dataparallel and not checkpoint_is_dataparallel:
                logging.info("Loading state_dict into DataParallel model.")
                new_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}
                model.load_state_dict(new_state_dict)
            elif not is_dataparallel and checkpoint_is_dataparallel:
                logging.info("Loading state_dict into single-GPU model.")
                new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(model_state_dict)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Check if a new learning rate is provided and update the optimizer
            if args.learning_rate and optimizer.param_groups[0]['lr'] != args.learning_rate:
                logging.info(f"Updating learning rate from checkpoint value {optimizer.param_groups[0]['lr']} to new value {args.learning_rate}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate

            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            if 'best_token_accuracy' in checkpoint:
                best_token_accuracy = checkpoint.get('best_token_accuracy', -1.0)

            
            logging.info(f"Resumed from epoch {checkpoint['epoch']}. Starting next epoch: {start_epoch}.")
            logging.info(f"Resumed best validation loss: {best_val_loss:.4f}")
            logging.info(f"Resumed best token accuracy: {best_token_accuracy:.4f}")
        else:
            logging.warning(f"Checkpoint file not found at {args.resume_from}. Starting training from scratch.")

    # --- 3. Training Loop ---
    logging.info("--- Starting Training ---")
    
    teacher_forcing_ratio = args.teacher_forcing_ratio

    import time
    for epoch in range(start_epoch, args.num_epochs):
        # Rebuild train_transform with new random augmentations for this epoch
        train_transform = build_train_transform()
        train_dataset.transform = train_transform
        # Log which augmentations are active/inactive
        all_aug_names = [name for name, _ in aug_transforms]
        active_set = set(active_aug_names)
        inactive_set = set(all_aug_names) - active_set
        logging.info(f"Epoch {epoch+1}: Active augmentations: {sorted(active_set)} | Inactive: {sorted(inactive_set)}")
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        # Tqdm writes to stderr by default. Changing to stdout.
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", file=sys.stdout)
        
        for batch in train_pbar:
            images = batch['image'].to(config.DEVICE)
            targets = batch['target'].to(config.DEVICE)
            
            # Ground truth is all but the first token (<sos>)
            ground_truth = targets[:, 1:]
            
            # Start with the <sos> token
            decoder_input_step = targets[:, 0].unsqueeze(1)
            
            loss = 0
            
            # Use teacher forcing or model's own predictions based on the ratio
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                if use_teacher_forcing:
                    # --- Teacher Forcing: Feed the entire ground truth sequence ---
                    decoder_input = targets[:, :-1]
                    output = model(images, decoder_input)
                    loss = criterion(output.reshape(-1, config.VOCAB_SIZE), ground_truth.reshape(-1))
                else:
                    # --- Scheduled Sampling: Feed one token at a time ---
                    batch_loss = 0
                    # The model inside DataParallel
                    model_instance = model.module if isinstance(model, nn.DataParallel) else model
                    
                    # Encode the image once
                    encoder_output = model_instance.encoder(images)
                    
                    for t in range(ground_truth.size(1)):
                        output_step = model_instance.decoder(decoder_input_step, encoder_output)
                        
                        # Calculate loss for this step
                        loss_step = criterion(output_step.squeeze(1), ground_truth[:, t])
                        batch_loss += loss_step
                        
                        # Get the prediction for the next step
                        _, topi = output_step.topk(1)
                        
                        # Detach from history to prevent backpropagating through the whole sequence
                        decoder_input_step = topi.detach()
                        
                    loss = batch_loss / ground_truth.size(1) # Average loss over sequence
            
            # Backward pass and optimization with Scaler
            scaler.scale(loss).backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                
                with torch.amp.autocast('cuda'):
                    output = model(images, decoder_input)
                    loss = criterion(output.reshape(-1, config.VOCAB_SIZE), ground_truth.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Decay teacher forcing ratio
        teacher_forcing_ratio *= args.teacher_forcing_decay
        logging.info(f"Teacher forcing ratio for next epoch: {teacher_forcing_ratio:.4f}")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Log the new learning rate
        new_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {new_lr}")

        # --- Qualitative Validation and Token Accuracy ---
        if fixed_val_image is not None:
            model.eval()
            with torch.no_grad():
                logging.info("--- Performing Qualitative Validation ---")
                
                # ALWAYS use the underlying model for prediction, not the DataParallel wrapper
                model_to_predict_with = model.module if isinstance(model, nn.DataParallel) else model
                
                # Create a temporary prediction image, adding a batch dimension if needed,
                # without modifying the original fixed_val_image.
                prediction_image = fixed_val_image.clone()
                if prediction_image.dim() == 3:
                    prediction_image = prediction_image.unsqueeze(0)

                predicted_smt = beam_search_predict(model_to_predict_with, prediction_image, tokenizer, beam_width=3, max_len=1024)
                
                # Calculate Token Accuracy
                gt_tokens = ground_truth_smt.split()
                pred_tokens = predicted_smt.split()
                correct_tokens = 0
                for i in range(min(len(gt_tokens), len(pred_tokens))):
                    if gt_tokens[i] == pred_tokens[i]:
                        correct_tokens += 1
                token_accuracy = correct_tokens / len(gt_tokens) if len(gt_tokens) > 0 else 0
                
                logging.info(f"    - Ground Truth SMT: {ground_truth_smt[:150]}...")
                logging.info(f"    - Prediction Sample:  {predicted_smt[:150]}...")
                logging.info(f"    - Token Accuracy: {token_accuracy:.4f}")

        # --- Save Checkpoint ---
        checkpoint_path = config.CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pth"
        
        # Save 'best' model based on token accuracy if available, otherwise fall back to val loss
        is_best = False
        if fixed_val_image is not None:
            if token_accuracy > best_token_accuracy:
                best_token_accuracy = token_accuracy
                is_best = True
        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_best = True

        if is_best:
            best_path = config.CHECKPOINT_DIR / "model_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_token_accuracy': best_token_accuracy,
                'args': args
            }, best_path)
            logging.info(f"Saved new best model to {best_path} (Epoch {epoch+1}, Token Accuracy: {token_accuracy:.4f})")

        # Save 'last' model (overwrite every epoch)
        last_path = config.CHECKPOINT_DIR / "model_last.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_token_accuracy': best_token_accuracy,
            'args': args
        }, last_path)

    logging.info("--- Training Complete ---")

if __name__ == '__main__':
    # Set the start method for multiprocessing
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()