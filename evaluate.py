import torch
import matplotlib.pyplot as plt
from model import UNet
from data_loader import WildfireDataset
import os
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_2.pth" # Load epoch 2
OUTPUT_IMG = "prediction_result.png"

def visualize_prediction(model, dataset, idx=0):
    model.eval()
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(DEVICE) # Add batch dim
    
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()
    
    # Convert back to numpy for plotting
    img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask = y.squeeze(0).cpu().numpy()
    pred = preds.squeeze(0).squeeze(0).cpu().numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title("Input Image")
    ax[0].axis("off")
    
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis("off")
    
    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")
    
    plt.savefig(OUTPUT_IMG)
    print(f"Saved visualization to {OUTPUT_IMG}")
    plt.close()

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Please run train.py first.")
        return

    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Use Kaggle CSV data for evaluation
    csv_path = "/Users/amogh/.cache/kagglehub/datasets/carlosparadis/fires-from-space-australia-and-new-zeland/versions/1/fire_archive_M6_96619.csv"
    if os.path.exists(csv_path):
        ds = WildfireDataset(csv_file=csv_path)
    else:
        ds = WildfireDataset(synthetic=True, n_synthetic=10)
    
    visualize_prediction(model, ds, idx=0)

if __name__ == "__main__":
    main()
