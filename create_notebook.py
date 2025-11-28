import json

cells = []

# Cell 1: Markdown Intro
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Wildfire Prediction with Satellite Imagery\n",
        "\n",
        "This notebook implements a U-Net model to predict wildfire probability from satellite imagery.\n",
        "It uses a hybrid approach: synthetic satellite imagery combined with real fire location data from Kaggle."
    ]
})

# Cell 2: Imports
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import Dataset, DataLoader, random_split\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from tqdm import tqdm\n",
        "\n",
        "# Check device\n",
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "print(f\"Using device: {DEVICE}\")"
    ]
})

# Cell 3: Dataset Class
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Dataset Loading & Generation\n",
        "The `WildfireDataset` class handles loading images or generating synthetic data based on real fire events from the Kaggle CSV."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class WildfireDataset(Dataset):\n",
        "    def __init__(self, csv_file=None, img_dir=None, mask_dir=None, transform=None, synthetic=False, n_synthetic=100):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            csv_file (string): Path to the Kaggle CSV file with fire data.\n",
        "            img_dir (string): Directory with all the images.\n",
        "            mask_dir (string): Directory with all the masks.\n",
        "            transform (callable, optional): Optional transform to be applied on a sample.\n",
        "            synthetic (bool): If True, generate fully random synthetic data (legacy mode).\n",
        "            n_synthetic (int): Number of synthetic samples to generate if csv_file is None.\n",
        "        \"\"\"\n",
        "        self.img_dir = img_dir\n",
        "        self.mask_dir = mask_dir\n",
        "        self.transform = transform\n",
        "        self.synthetic = synthetic\n",
        "        self.csv_data = None\n",
        "        \n",
        "        if csv_file:\n",
        "            print(f\"Loading fire data from {csv_file}...\")\n",
        "            self.csv_data = pd.read_csv(csv_file)\n",
        "            # Filter for high confidence fires to ensure quality training data\n",
        "            if 'confidence' in self.csv_data.columns:\n",
        "                 self.csv_data = self.csv_data[self.csv_data['confidence'] > 50]\n",
        "            print(f\"Loaded {len(self.csv_data)} fire events.\")\n",
        "        \n",
        "        self.n_synthetic = n_synthetic\n",
        "        \n",
        "        if not self.synthetic and self.csv_data is None:\n",
        "            if img_dir is None or mask_dir is None:\n",
        "                raise ValueError(\"img_dir and mask_dir must be provided if synthetic is False and no CSV is provided\")\n",
        "            self.images = sorted(os.listdir(img_dir))\n",
        "            self.masks = sorted(os.listdir(mask_dir))\n",
        "\n",
        "    def __len__(self):\n",
        "        if self.csv_data is not None:\n",
        "            return len(self.csv_data)\n",
        "        if self.synthetic:\n",
        "            return self.n_synthetic\n",
        "        return len(self.images)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        if self.csv_data is not None:\n",
        "            # Use real data to drive synthetic generation\n",
        "            row = self.csv_data.iloc[idx]\n",
        "            frp = row['frp'] if 'frp' in row else 10.0\n",
        "            \n",
        "            # Normalize FRP to a reasonable radius for the blob\n",
        "            # FRP typically ranges from 0 to 1000+, but most are < 100.\n",
        "            # We'll map it to a radius between 5 and 60 pixels.\n",
        "            radius = int(np.clip(frp / 2.0, 5, 60))\n",
        "            \n",
        "            # Generate synthetic image/mask based on this \"real\" fire property\n",
        "            image = np.random.rand(256, 256, 3).astype(np.float32)\n",
        "            mask = np.zeros((256, 256), dtype=np.float32)\n",
        "            \n",
        "            # Center the fire for simplicity, or random offset\n",
        "            cx, cy = np.random.randint(50, 206, 2)\n",
        "            \n",
        "            y, x = np.ogrid[:256, :256]\n",
        "            mask_area = ((x - cx)**2 + (y - cy)**2 <= radius**2)\n",
        "            mask[mask_area] = 1.0\n",
        "            \n",
        "            # Add fire colors\n",
        "            image[mask_area, 0] = np.clip(image[mask_area, 0] + 0.6, 0, 1) # Red\n",
        "            image[mask_area, 1] = np.clip(image[mask_area, 1] + 0.3, 0, 1) # Green\n",
        "            \n",
        "            image = torch.from_numpy(image).permute(2, 0, 1).float()\n",
        "            mask = torch.from_numpy(mask).unsqueeze(0).float()\n",
        "            \n",
        "            return image, mask\n",
        "\n",
        "        elif self.synthetic:\n",
        "            # Fully random synthetic data\n",
        "            image = np.random.rand(256, 256, 3).astype(np.float32)\n",
        "            mask = np.zeros((256, 256), dtype=np.float32)\n",
        "            cx, cy = np.random.randint(50, 200, 2)\n",
        "            r = np.random.randint(10, 50)\n",
        "            y, x = np.ogrid[:256, :256]\n",
        "            mask_area = ((x - cx)**2 + (y - cy)**2 <= r**2)\n",
        "            mask[mask_area] = 1.0\n",
        "            image[mask_area, 0] = np.clip(image[mask_area, 0] + 0.5, 0, 1)\n",
        "            image[mask_area, 1] = np.clip(image[mask_area, 1] + 0.2, 0, 1)\n",
        "            image = torch.from_numpy(image).permute(2, 0, 1).float()\n",
        "            mask = torch.from_numpy(mask).unsqueeze(0).float()\n",
        "            return image, mask\n",
        "            \n",
        "        else:\n",
        "            img_path = os.path.join(self.img_dir, self.images[idx])\n",
        "            mask_path = os.path.join(self.mask_dir, self.masks[idx])\n",
        "            image = np.array(Image.open(img_path).convert(\"RGB\"))\n",
        "            mask = np.array(Image.open(mask_path).convert(\"L\"))\n",
        "            image = image / 255.0\n",
        "            mask = mask / 255.0\n",
        "            mask = (mask > 0.5).astype(np.float32)\n",
        "            image = torch.from_numpy(image).permute(2, 0, 1).float()\n",
        "            mask = torch.from_numpy(mask).unsqueeze(0).float()\n",
        "            return image, mask"
    ]
})

# Cell 4: Model
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. U-Net Model Architecture\n",
        "Standard U-Net implementation for semantic segmentation."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class DoubleConv(nn.Module):\n",
        "    \"\"\"(convolution => [BN] => ReLU) * 2\"\"\"\n",
        "\n",
        "    def __init__(self, in_channels, out_channels, mid_channels=None):\n",
        "        super().__init__()\n",
        "        if not mid_channels:\n",
        "            mid_channels = out_channels\n",
        "        self.double_conv = nn.Sequential(\n",
        "            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),\n",
        "            nn.BatchNorm2d(mid_channels),\n",
        "            nn.ReLU(inplace=True),\n",
        "            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),\n",
        "            nn.BatchNorm2d(out_channels),\n",
        "            nn.ReLU(inplace=True)\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.double_conv(x)\n",
        "\n",
        "\n",
        "class Down(nn.Module):\n",
        "    \"\"\"Downscaling with maxpool then double conv\"\"\"\n",
        "\n",
        "    def __init__(self, in_channels, out_channels):\n",
        "        super().__init__()\n",
        "        self.maxpool_conv = nn.Sequential(\n",
        "            nn.MaxPool2d(2),\n",
        "            DoubleConv(in_channels, out_channels)\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.maxpool_conv(x)\n",
        "\n",
        "\n",
        "class Up(nn.Module):\n",
        "    \"\"\"Upscaling then double conv\"\"\"\n",
        "\n",
        "    def __init__(self, in_channels, out_channels, bilinear=True):\n",
        "        super().__init__()\n",
        "\n",
        "        # if bilinear, use the normal convolutions to reduce the number of channels\n",
        "        if bilinear:\n",
        "            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)\n",
        "            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)\n",
        "        else:\n",
        "            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)\n",
        "            self.conv = DoubleConv(in_channels, out_channels)\n",
        "\n",
        "    def forward(self, x1, x2):\n",
        "        x1 = self.up(x1)\n",
        "        # input is CHW\n",
        "        diffY = x2.size()[2] - x1.size()[2]\n",
        "        diffX = x2.size()[3] - x1.size()[3]\n",
        "\n",
        "        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,\n",
        "                        diffY // 2, diffY - diffY // 2])\n",
        "        x = torch.cat([x2, x1], dim=1)\n",
        "        return self.conv(x)\n",
        "\n",
        "\n",
        "class OutConv(nn.Module):\n",
        "    def __init__(self, in_channels, out_channels):\n",
        "        super(OutConv, self).__init__()\n",
        "        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.conv(x)\n",
        "\n",
        "\n",
        "class UNet(nn.Module):\n",
        "    def __init__(self, n_channels, n_classes, bilinear=True):\n",
        "        super(UNet, self).__init__()\n",
        "        self.n_channels = n_channels\n",
        "        self.n_classes = n_classes\n",
        "        self.bilinear = bilinear\n",
        "\n",
        "        self.inc = DoubleConv(n_channels, 64)\n",
        "        self.down1 = Down(64, 128)\n",
        "        self.down2 = Down(128, 256)\n",
        "        self.down3 = Down(256, 512)\n",
        "        factor = 2 if bilinear else 1\n",
        "        self.down4 = Down(512, 1024 // factor)\n",
        "        self.up1 = Up(1024, 512 // factor, bilinear)\n",
        "        self.up2 = Up(512, 256 // factor, bilinear)\n",
        "        self.up3 = Up(256, 128 // factor, bilinear)\n",
        "        self.up4 = Up(128, 64, bilinear)\n",
        "        self.outc = OutConv(64, n_classes)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x1 = self.inc(x)\n",
        "        x2 = self.down1(x1)\n",
        "        x3 = self.down2(x2)\n",
        "        x4 = self.down3(x3)\n",
        "        x5 = self.down4(x4)\n",
        "        x = self.up1(x5, x4)\n",
        "        x = self.up2(x, x3)\n",
        "        x = self.up3(x, x2)\n",
        "        x = self.up4(x, x1)\n",
        "        logits = self.outc(x)\n",
        "        return logits"
    ]
})

# Cell 5: Training Setup
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Training Loop\n",
        "Define hyperparameters and the training function."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Hyperparameters\n",
        "LEARNING_RATE = 1e-4\n",
        "BATCH_SIZE = 8\n",
        "NUM_EPOCHS = 2 # Reduced for demo\n",
        "\n",
        "def train_fn(loader, model, optimizer, loss_fn, scaler):\n",
        "    loop = tqdm(loader)\n",
        "    total_loss = 0\n",
        "    \n",
        "    for batch_idx, (data, targets) in enumerate(loop):\n",
        "        data = data.to(DEVICE)\n",
        "        targets = targets.to(DEVICE)\n",
        "\n",
        "        # Forward\n",
        "        predictions = model(data)\n",
        "        loss = loss_fn(predictions, targets)\n",
        "\n",
        "        # Backward\n",
        "        optimizer.zero_grad()\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        # Update tqdm loop\n",
        "        total_loss += loss.item()\n",
        "        loop.set_postfix(loss=loss.item())\n",
        "        \n",
        "    return total_loss / len(loader)"
    ]
})

# Cell 6: Evaluation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Evaluation & Visualization"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def visualize_prediction(model, dataset, idx=0):\n",
        "    model.eval()\n",
        "    x, y = dataset[idx]\n",
        "    x = x.unsqueeze(0).to(DEVICE) # Add batch dim\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        preds = torch.sigmoid(model(x))\n",
        "        preds = (preds > 0.5).float()\n",
        "    \n",
        "    # Convert back to numpy for plotting\n",
        "    img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()\n",
        "    mask = y.squeeze(0).cpu().numpy()\n",
        "    pred = preds.squeeze(0).squeeze(0).cpu().numpy()\n",
        "    \n",
        "    fig, ax = plt.subplots(1, 3, figsize=(15, 5))\n",
        "    ax[0].imshow(img)\n",
        "    ax[0].set_title(\"Input Image\")\n",
        "    ax[0].axis(\"off\")\n",
        "    \n",
        "    ax[1].imshow(mask, cmap=\"gray\")\n",
        "    ax[1].set_title(\"Ground Truth Mask\")\n",
        "    ax[1].axis(\"off\")\n",
        "    \n",
        "    ax[2].imshow(pred, cmap=\"gray\")\n",
        "    ax[2].set_title(\"Predicted Mask\")\n",
        "    ax[2].axis(\"off\")\n",
        "    \n",
        "    plt.show()"
    ]
})

# Cell 7: Main Execution
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Run Training and Evaluation"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Initialize model\n",
        "model = UNet(n_channels=3, n_classes=1).to(DEVICE)\n",
        "loss_fn = nn.BCEWithLogitsLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)\n",
        "scaler = torch.cuda.amp.GradScaler() if DEVICE == \"cuda\" else None\n",
        "\n",
        "# Load Data\n",
        "csv_path = \"/Users/amogh/.cache/kagglehub/datasets/carlosparadis/fires-from-space-australia-and-new-zeland/versions/1/fire_archive_M6_96619.csv\"\n",
        "if os.path.exists(csv_path):\n",
        "    print(\"Found Kaggle CSV.\")\n",
        "    full_dataset = WildfireDataset(csv_file=csv_path)\n",
        "else:\n",
        "    print(\"CSV not found, falling back to pure synthetic.\")\n",
        "    full_dataset = WildfireDataset(synthetic=True, n_synthetic=200)\n",
        "\n",
        "# Split into train/val\n",
        "train_size = int(0.8 * len(full_dataset))\n",
        "val_size = len(full_dataset) - train_size\n",
        "train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])\n",
        "\n",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
        "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
        "\n",
        "print(f\"Training on {DEVICE} with {len(train_dataset)} samples.\")\n",
        "\n",
        "# Train\n",
        "for epoch in range(NUM_EPOCHS):\n",
        "    print(f\"Epoch {epoch+1}/{NUM_EPOCHS}\")\n",
        "    model.train()\n",
        "    avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)\n",
        "    print(f\"Average Loss: {avg_loss:.4f}\")\n",
        "\n",
        "print(\"Training complete.\")\n",
        "\n",
        "# Visualize\n",
        "visualize_prediction(model, full_dataset, idx=0)"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("wildfire_prediction.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
