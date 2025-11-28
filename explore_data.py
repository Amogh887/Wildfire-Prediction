import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("carlosparadis/fires-from-space-australia-and-new-zeland")

print("Path to dataset files:", path)

# List files
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
