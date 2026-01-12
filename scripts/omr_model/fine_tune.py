import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

from . import config
from .dataset import ScoreDataset
from .tokenizer import StTokenizer
from .model import ImageToStModel

def collate_fn(batch, pad_token_id):
    images = torch.stack([item['image'] for item in batch])
    encoded_strings = [item['encoded_st'] for item in batch]
    max_len = max(len(s) for s in encoded_strings)
    padded_strings = []
    for s in encoded_strings:
        padded = s + [pad_token_id] * (max_len - len(s))
        padded_strings.append(torch.tensor(padded, dtype=torch.long))
    targets = torch.stack(padded_strings)
    return {'image': images, 'target': targets}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the OMR model.")
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=config.FINE_LEARNING_RATE, help='Lower LR for fine-tuning')
    parser.add_argument('--num-epochs', type=int, default=config.FINE_NUM_EPOCHS)
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)
    parser.add_argument('--validation-split', type=float, default=config.VALIDATION_SPLIT)
    parser.add_argument('--pretrained-checkpoint', type=str, default=str(config.FINETUNE_PRETRAINED_MODEL_PATH), help='Path to pre-trained model .pth')
    parser.add_argument('--finetune-dataset', type=str, required=True, help='Path to fine-tune dataset manifest (json)')
    parser.add_argument('--tokenizer-vocab', type=str, default=str(config.TOKENIZER_VOCAB_PATH))
    parser.add_argument('--output-dir', type=str, default=str(config.CHECKPOINT_DIR))
    return parser.parse_args()

def main():
    args = parse_args()
    device = config.DEVICE
    print(f"Fine-tuning on device: {device}")

    # Load tokenizer
    tokenizer = StTokenizer()
    tokenizer.load(args.tokenizer_vocab)
    pad_token_id = tokenizer.token_to_id['<pad>']

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ScoreDataset(
        manifest_path=args.finetune_dataset,
        tokenizer=tokenizer,
        transform=transform
    )

    # Data splitting
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=lambda b: collate_fn(b, pad_token_id)
    )

    # Model
    model = ImageToStModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)

    # Load pre-trained weights
    print(f"Loading pre-trained weights from {args.pretrained_checkpoint}")
    model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=device))

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Fine-tuning loop
    print("\n--- Starting Fine-tuning ---")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch in train_pbar:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            decoder_input = targets[:, :-1]
            ground_truth = targets[:, 1:]
            optimizer.zero_grad()
            output = model(images, decoder_input)
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), ground_truth.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    targets = batch['target'].to(device)
                    decoder_input = targets[:, :-1]
                    ground_truth = targets[:, 1:]
                    output = model(images, decoder_input)
                    loss = criterion(output.reshape(-1, tokenizer.vocab_size), ground_truth.reshape(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {avg_train_loss:.4f} (No validation)")

        # Save checkpoint
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"finetuned_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

    print("\n--- Fine-tuning Complete ---")

if __name__ == '__main__':
    main()