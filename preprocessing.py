import os
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

OPTIMIZED_DIR = './data/optimized'

class ImageFolderForBinaryClassification(ImageFolder):
    def __init__(self, root, target, transform=None,):
        super().__init__(root, transform=transform)
        self.target = target
        label_mapping = map_labels(root, target)
        self.__pre_process_data(root, label_mapping)

    def __getitem__(self, index):
        # Get the default ImageFolder result (image, label)
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __pre_process_data(self,root, label_mapping):
        for i in range(len(self.samples)):
            path, _ = self.samples[i]
            label = label_mapping.get(path, -1) # Return -1 if no label found
            self.samples[i] = (path, label)

    # probably redundant
    def get_subset_by_indices(self, indices):
        """
        Returns a subset of the dataset using the specified indices.
        """
        subset = ImageFolderForBinaryClassification(self.root, self.target, transform=self.transform)
        subset.samples = [subset.samples[i] for i in indices ]
        return subset


def map_labels(root, target):
    classes_df = pd.read_csv(os.path.join(root, 'classes.csv'))  # Read CSV file
    add_backslash= lambda s: s.replace('/','\\')
    label_mapping = { f"{root}\\{add_backslash(row['filename'])}": row[target] for _, row in classes_df.iterrows()}  # Map image names to labels
    return label_mapping


class NumPyDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.images = data["images"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def optimize_dataset(dataset, output_name):
    # Step 2: Convert and Save the Dataset as NumPy Arrays
    images = []
    labels = []
    print(f"Converting {len(dataset)} images into NumPy format...")
    for data, label in dataset:
        images.append(data.numpy())  # Convert Tensor to NumPy
        labels.append(label)
    path = f"{OPTIMIZED_DIR}/{output_name}.npz"
    np.savez_compressed(path, images=np.array(images), labels=np.array(labels))
    print(f"Saved dataset to {path}")

