import os
import torch
import numpy as np
from PIL import Image

from torchvision import transforms

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight',
    #4: 'late_blight',
}

class FieldTomatoImages(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Root directory containing subfolders for each class
            transform (callable, optional): Optional transform to apply to each image
        """
        self.root = root
        self.transform = transform
        
        self.img_files = []
        self.labels = []

        # Reverse map: class name -> label
        class_to_label = {v: k for k, v in classes.items()}

        # Iterate over each class folder
        for class_name, label in class_to_label.items():
            class_folder = os.path.join(root, class_name)
            if not os.path.exists(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.endswith(".jpg"):
                    self.img_files.append(os.path.join(class_folder, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        img = transforms.functional.rotate(img, -90)
        
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
