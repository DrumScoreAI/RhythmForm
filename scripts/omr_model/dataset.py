import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from . import config

# --- Path Configuration ---
# Use the RHYTHMFORMHOME env var for the project root, with a fallback.
# This ensures the path is correct regardless of where the script is run from.
PROJECT_ROOT = config.PROJECT_ROOT
DATASET_JSON_PATH = config.DATASET_JSON_PATH


class ScoreDataset(Dataset):
    """
    A PyTorch Dataset for loading score images and their corresponding ST strings.
    """
    def __init__(self, manifest_path=DATASET_JSON_PATH, tokenizer=None, transform=None):
        """
        Args:
            manifest_path (str or Path): Path to the dataset.json manifest file.
            tokenizer (StTokenizer, optional): The tokenizer to use for encoding strings.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.manifest_path = Path(manifest_path)
        self.root_dir = self.manifest_path.parent
        self.tokenizer = tokenizer
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}")

        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def set_transform(self, transform):
        """
        Set the transform for the dataset.
        """
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            new_manifest = self.manifest[idx]
            new_dataset = self.__class__(
                manifest_path=self.manifest_path,
                tokenizer=self.tokenizer,
                transform=self.transform
            )
            new_dataset.manifest = new_manifest
            return new_dataset

        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.manifest[idx]
        smt_string = item['smt_string']
        # Fix: If image_path is absolute, use as is. If it starts with 'fine_tuning/', strip it.
        image_path_str = item['image_path']
        if image_path_str.startswith('fine_tuning/'):
            image_path_str = image_path_str[len('fine_tuning/'):]  # Remove leading 'fine_tuning/'
        image_path = self.root_dir / image_path_str
        
        try:
            image = Image.open(image_path).convert('L') # Convert to grayscale
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            raise 

        # --- THIS IS THE FIX ---
        # The transform must be applied to the image object itself,
        # not to the dictionary that contains it.
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'smt_string': smt_string}

        if self.tokenizer:
            sample['encoded_smt'] = self.tokenizer.encode(smt_string)

        return sample

# This block allows you to test the dataset by running `python omr_model/dataset.py`
if __name__ == '__main__':
    print(f"Loading dataset from: {DATASET_JSON_PATH}")
    
    # Create an instance of the dataset
    score_dataset = ScoreDataset(manifest_path=DATASET_JSON_PATH)
    
    # Check the total number of samples
    print(f"Dataset contains {len(score_dataset)} samples.")
    
    # Get and inspect the first sample
    if len(score_dataset) > 0:
        first_sample = score_dataset[0]
        print("\n--- Inspecting first sample ---")
        print(f"Image tensor shape: {first_sample['image'].shape}")
        print(f"Image tensor dtype: {first_sample['image'].dtype}")
        print(f"ST string (first 80 chars): {first_sample['smt_string'][:80]}...")
        print("-----------------------------\n")
        print("Dataset class seems to be working correctly!")
    else:
        print("Dataset is empty. Cannot inspect a sample.")
