import os
import pandas as pd
from torchvision.datasets import ImageFolder


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
            #path = os.path.relpath(path, root).replace('\\', '/')
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

