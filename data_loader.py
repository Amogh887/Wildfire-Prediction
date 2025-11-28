import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

class WildfireDataset(Dataset):
    def __init__(self, csv_file=None, img_dir=None, mask_dir=None, transform=None, synthetic=False, n_synthetic=100):
        """
        Args:
            csv_file (string): Path to the Kaggle CSV file with fire data.
            img_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            synthetic (bool): If True, generate fully random synthetic data (legacy mode).
            n_synthetic (int): Number of synthetic samples to generate if csv_file is None.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.synthetic = synthetic
        self.csv_data = None
        
        if csv_file:
            print(f"Loading fire data from {csv_file}...")
            self.csv_data = pd.read_csv(csv_file)
            # Filter for high confidence fires to ensure quality training data
            if 'confidence' in self.csv_data.columns:
                 self.csv_data = self.csv_data[self.csv_data['confidence'] > 50]
            print(f"Loaded {len(self.csv_data)} fire events.")
        
        self.n_synthetic = n_synthetic
        
        if not self.synthetic and self.csv_data is None:
            if img_dir is None or mask_dir is None:
                raise ValueError("img_dir and mask_dir must be provided if synthetic is False and no CSV is provided")
            self.images = sorted(os.listdir(img_dir))
            self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        if self.csv_data is not None:
            return len(self.csv_data)
        if self.synthetic:
            return self.n_synthetic
        return len(self.images)

    def __getitem__(self, idx):
        if self.csv_data is not None:
            # Use real data to drive synthetic generation
            row = self.csv_data.iloc[idx]
            frp = row['frp'] if 'frp' in row else 10.0
            
            # Normalize FRP to a reasonable radius for the blob
            # FRP typically ranges from 0 to 1000+, but most are < 100.
            # We'll map it to a radius between 5 and 60 pixels.
            radius = int(np.clip(frp / 2.0, 5, 60))
            
            # Generate synthetic image/mask based on this "real" fire property
            image = np.random.rand(256, 256, 3).astype(np.float32)
            mask = np.zeros((256, 256), dtype=np.float32)
            
            # Center the fire for simplicity, or random offset
            cx, cy = np.random.randint(50, 206, 2)
            
            y, x = np.ogrid[:256, :256]
            mask_area = ((x - cx)**2 + (y - cy)**2 <= radius**2)
            mask[mask_area] = 1.0
            
            # Add fire colors
            image[mask_area, 0] = np.clip(image[mask_area, 0] + 0.6, 0, 1) # Red
            image[mask_area, 1] = np.clip(image[mask_area, 1] + 0.3, 0, 1) # Green
            
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            
            return image, mask

        elif self.synthetic:
            # Fully random synthetic data
            image = np.random.rand(256, 256, 3).astype(np.float32)
            mask = np.zeros((256, 256), dtype=np.float32)
            cx, cy = np.random.randint(50, 200, 2)
            r = np.random.randint(10, 50)
            y, x = np.ogrid[:256, :256]
            mask_area = ((x - cx)**2 + (y - cy)**2 <= r**2)
            mask[mask_area] = 1.0
            image[mask_area, 0] = np.clip(image[mask_area, 0] + 0.5, 0, 1)
            image[mask_area, 1] = np.clip(image[mask_area, 1] + 0.2, 0, 1)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            return image, mask
            
        else:
            img_path = os.path.join(self.img_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            image = image / 255.0
            mask = mask / 255.0
            mask = (mask > 0.5).astype(np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            return image, mask

if __name__ == "__main__":
    # Test with a dummy CSV if available, or just synthetic
    ds = WildfireDataset(synthetic=True, n_synthetic=5)
    img, mask = ds[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
