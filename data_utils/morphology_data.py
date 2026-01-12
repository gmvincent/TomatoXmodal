import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}

class MorphologyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Directory containing .xlsx file
            transform (callable, optional): Optional transform to apply to each voxel
        """
        data_file = pd.read_excel(os.path.join(root, "tomato_features.xlsx"))
        
        data_file['timestamp'] = pd.to_datetime(data_file['timestamp'])
        data_file = data_file[data_file['timestamp'] <= datetime(2025, 5, 1)]
        data_file = data_file[data_file["treatment"] != 1]
        
        cols = ["Digital biomass [mmÂ³]", "Height [mm]", "greenness average", "hue average [Â°]", "Leaf angle [Â°]", "Leaf area [mmÂ²]", "Leaf area index [mmÂ²/mmÂ²]", "Light penetration depth [mm]", "NDVI average"]
        
        X_raw = data_file[cols].values.astype(np.float32)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)

        self.data = torch.tensor(X_scaled, dtype=torch.float32)
        self.labels = torch.tensor(data_file['treatment'].values, dtype=torch.long)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y