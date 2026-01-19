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
from .tokenizer import SmtTokenizer
from .model import ImageToSmtModel
from .predict import beam_search_predict

def collate_fn(batch, pad_token_id):
    images = torch.stack([item['image'] for item in batch])
    encoded_strings = [item['encoded_smt'] for item in batch]
    max_len = max(len(s) for s in encoded_strings)
    padded_strings = []
    for s in encoded_strings:
        padded = s + [pad_token_id] * (max_len - len(s))
        padded_strings.append(torch.tensor(padded, dtype=torch.long))
    targets = torch.stack(padded_strings)
    return {'image': images, 'target': targets}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the OMR model.")
    parser.add_argument('--batch-size', type=int, default=128)
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
    tokenizer = SmtTokenizer()
    tokenizer.load(args.tokenizer_vocab)
    pad_token_id = tokenizer.token_to_id['<pad>']

    # Augmentation transforms
    aug_transforms = [
        ("RandomAffine", transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5)),
        ("ElasticTransform", transforms.RandomApply([transforms.ElasticTransform(alpha=75.0, sigma=5.0)], p=0.5)),
        ("RandomPerspective", transforms.RandomPerspective(distortion_scale=0.2, p=0.5)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.4, contrast=0.4)),
    ]

    active_aug_names = []
    active_aug_transforms = []
    def build_train_transform():
        import random
        selected = random.sample(aug_transforms, 2)
        nonlocal active_aug_names, active_aug_transforms
        active_aug_names = [name for name, _ in selected]
        active_aug_transforms = [t for _, t in selected]
        return transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            *active_aug_transforms,
            transforms.ToTensor()
        ])

    train_transform = build_train_transform()
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor()
    ])

    dataset = ScoreDataset(
        manifest_path=args.finetune_dataset,
        tokenizer=tokenizer,
        transform=train_transform
    )

    # Data splitting
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.validation_split * dataset_size))
    np.random.shuffle(indices)
    val_indices = indices[:split]
    train_indices = indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Grab a single, fixed image from the validation set for qualitative assessment
    val_dataset_for_prediction = ScoreDataset(
        manifest_path=args.finetune_dataset,
        tokenizer=tokenizer,
        transform=val_transform  # Use the non-augmenting transform
    )
    if val_indices:
        # Get the dictionary for the first validation sample
        val_sample = val_dataset_for_prediction[val_indices[0]]
        # Extract the image tensor and move it to the correct device
        fixed_val_image = val_sample['image'].to(device)
        # Decode the ground truth SMT string for comparison
        ground_truth_smt = tokenizer.decode(val_sample['encoded_smt'])
        print(f"\nUsing validation image index {val_indices[0]} for qualitative prediction checks.")
    else:
        fixed_val_image = None
        ground_truth_smt = None
        print("\nNo validation set found, skipping qualitative prediction checks.")


    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=lambda b: collate_fn(b, pad_token_id)
    )

    # Model
    model = ImageToSmtModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)

    # Load pre-trained weights
    print(f"Loading pre-trained weights from {args.pretrained_checkpoint}")
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Fine-tuning loop
    print("\n--- Starting Fine-tuning ---")

    # Determine starting epoch for checkpoint naming to allow for resuming
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_files = list(output_dir.glob("*epoch_*.pth"))
    starting_epoch_offset = 0
    if epoch_files:
        starting_epoch_offset = max([int(f.stem.split("_")[-1]) for f in epoch_files])
    if starting_epoch_offset > 0:
        print(f"Resuming. Found existing checkpoints. Starting epoch numbering from {starting_epoch_offset + 1}.")

    import time
    for epoch in range(args.num_epochs):
        # Rebuild train_transform with new random augmentations for this epoch
        train_transform = build_train_transform()
        dataset.transform = train_transform
        
        current_epoch_num = epoch + 1 + starting_epoch_offset
        total_epochs = args.num_epochs + starting_epoch_offset

        # Log which augmentations are active/inactive
        all_aug_names = [name for name, _ in aug_transforms]
        active_set = set(active_aug_names)
        inactive_set = set(all_aug_names) - active_set
        print(f"Epoch {current_epoch_num}: Active augmentations: {sorted(active_set)} | Inactive: {sorted(inactive_set)}")

        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {current_epoch_num}/{total_epochs} [Train]")
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
            print(f"Epoch {current_epoch_num}/{total_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {current_epoch_num}/{total_epochs} -> Train Loss: {avg_train_loss:.4f} (No validation)")
        
        # --- Qualitative Validation ---
        if fixed_val_image is not None:
            model.eval()
            with torch.no_grad():
                predicted_smt = beam_search_predict(model, fixed_val_image, tokenizer, beam_width=3, max_len=200)
                print(f"    - Ground Truth SMT: {ground_truth_smt[:150]}...")
                print(f"    - Prediction Sample:  {predicted_smt[:150]}...")


        # Save checkpoint
        checkpoint_path = output_dir / f"finetuned_model_epoch_{current_epoch_num}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        # Save 'last' model (overwrite every epoch)
        last_path = output_dir / "finetuned_model_last.pth"
        torch.save(model.state_dict(), last_path)

        # Save 'best' model (lowest validation loss)
        if epoch == 0:
            best_val_loss = avg_val_loss if len(val_loader) > 0 else avg_train_loss
            best_path = output_dir / "finetuned_model_best.pth"
            torch.save(model.state_dict(), best_path)
        else:
            if len(val_loader) > 0 and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = output_dir / "finetuned_model_best.pth"
                torch.save(model.state_dict(), best_path)

    print("\n--- Fine-tuning Complete ---")

if __name__ == '__main__':
    main()