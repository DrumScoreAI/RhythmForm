import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import all our custom modules
from . import config
from .dataset import ScoreDataset
from .tokenizer import SmtTokenizer
from .model import ImageToSmtModel

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

def main():
    """
    Main training function.
    """
    # --- 1. Setup ---
    print(f"Using device: {config.DEVICE}")

    # Load tokenizer
    tokenizer = SmtTokenizer()
    tokenizer.load(config.TOKENIZER_VOCAB_PATH)
    pad_token_id = tokenizer.token_to_id['<pad>']
    
    # Load dataset
    full_dataset = ScoreDataset(manifest_path=config.DATASET_JSON_PATH)
    
    # Pre-tokenize all strings in the dataset (more efficient)
    for i in range(len(full_dataset.data)):
        smt_string = full_dataset.data[i]['smt_string']
        full_dataset.data[i]['encoded_smt'] = tokenizer.encode(smt_string)

    # Split dataset into training and validation
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=4  # Use 4 CPU cores to pre-load data
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )

    # --- 2. Model, Loss, and Optimizer ---
    model = ImageToSmtModel(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        image_height=config.IMG_HEIGHT,
        image_width=config.IMG_WIDTH,
        patch_size=config.PATCH_SIZE
    ).to(config.DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss function ignores the padding token
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- 3. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for batch in train_pbar:
            images = batch['image'].to(config.DEVICE)
            targets = batch['target'].to(config.DEVICE)
            
            # Decoder input is all but the last token (<eos>)
            # Ground truth is all but the first token (<sos>)
            decoder_input = targets[:, :-1]
            ground_truth = targets[:, 1:]
            
            # Forward pass
            optimizer.zero_grad()
            output = model(images, decoder_input)
            
            # Reshape for loss calculation
            # Output: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # Ground Truth: (batch, seq_len) -> (batch * seq_len)
            loss = criterion(output.reshape(-1, config.VOCAB_SIZE), ground_truth.reshape(-1))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(config.DEVICE)
                targets = batch['target'].to(config.DEVICE)
                
                decoder_input = targets[:, :-1]
                ground_truth = targets[:, 1:]
                
                output = model(images, decoder_input)
                loss = criterion(output.reshape(-1, config.VOCAB_SIZE), ground_truth.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Save Checkpoint ---
        checkpoint_path = config.CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

    print("\n--- Training Complete ---")


if __name__ == '__main__':
    main()