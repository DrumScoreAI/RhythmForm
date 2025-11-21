import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# We'll move these to a config.py file later, but for now, it's useful to have them here.
# Assumes SFHOME is set, otherwise defaults to the current project structure.
PROJECT_ROOT = Path(os.environ.get('SFHOME', Path(__file__).parent.parent))
DATASET_JSON_PATH = PROJECT_ROOT / 'training_data' / 'dataset.json'

class ScoreDataset(Dataset):
    """
    PyTorch Dataset for loading score images and their SMT string labels.
    """
    def __init__(self, manifest_path, image_transform=None):
        """
        Args:
            manifest_path (str or Path): Path to the dataset.json manifest file.
            image_transform (callable, optional): A torchvision transform to be applied on an image.
        """
        self.manifest_path = manifest_path
        
        # If no transform is provided, use a default one.
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((1024, 512)), # Example size, we will configure this
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = image_transform

        with open(self.manifest_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the image tensor and the SMT string.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]
        
        # The paths in dataset.json are relative to the project root.
        image_path = PROJECT_ROOT / item['image_path']
        smt_string = item['smt_string']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            # Return dummy data or raise an error
            return None, None

        if self.image_transform:
            image = self.image_transform(image)
            
        sample = {'image': image, 'smt_string': smt_string}

        # If the pre-encoded string exists (added by train.py), include it.
        if 'encoded_smt' in item:
            sample['encoded_smt'] = item['encoded_smt']

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
        print(f"SMT string (first 80 chars): {first_sample['smt_string'][:80]}...")
        print("-----------------------------\n")
        print("Dataset class seems to be working correctly!")
    else:
        print("Dataset is empty. Cannot inspect a sample.")
