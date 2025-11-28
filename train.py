import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import UNet
from data_loader import WildfireDataset
import os
from tqdm import tqdm

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        # with torch.cuda.amp.autocast(): # Mixed precision if available
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # Update tqdm loop
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Initialize model
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # Combines Sigmoid and BCELoss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    # Load Data
    # Using Kaggle CSV data to drive synthetic generation
    csv_path = "/Users/amogh/.cache/kagglehub/datasets/carlosparadis/fires-from-space-australia-and-new-zeland/versions/1/fire_archive_M6_96619.csv"
    if os.path.exists(csv_path):
        full_dataset = WildfireDataset(csv_file=csv_path)
    else:
        print("CSV not found, falling back to pure synthetic.")
        full_dataset = WildfireDataset(synthetic=True, n_synthetic=200)
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {DEVICE} with {len(train_dataset)} samples.")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

    print("Training complete.")

if __name__ == "__main__":
    main()
