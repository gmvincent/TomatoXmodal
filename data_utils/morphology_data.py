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

reg_scale = [
    1.0,  # <1%,    score=1
    3.0,  # 1-3%,   score=2
    6.0,  # 3-6%,   score=3
    12.0, # 6-12%,  score=4
    25.0, # 12-25%, score=5
    50.0, # 25-50%, score=6
    75.0, # 50-75%, score=7
    87.0, # 75-87%, score=8
    94.0, # 87-94%, score=9
    97.0, # 94-97%, score=10
]

class MorphologyDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            root: str,  
            split: str, 
            transform=None,
            num_tasks: str="1", 
            task: str="classification",
            reg_target: str="hb", # "hb" or "pct
            ):
        """
        Args:
            root (str): directory containing image files
            split (str): train-val-test split subdirectory
            transform (callable, optional): Optional transform to apply to each voxel
        """
        self.num_tasks = num_tasks
        self.tasks = [task] if isinstance(task, str) else list(task)
        
        data_file = pd.read_excel(os.path.join(root, f"{split}_tomato_features.xlsx"))
        
        data_file['timestamp'] = pd.to_datetime(data_file['timestamp'])
        data_file = data_file[data_file['timestamp'] <= datetime(2025, 5, 1)]
        data_file = data_file[data_file["treatment"] != 1]
        
        cols = ["Digital biomass [mmÂ³]", "Height [mm]", "greenness average", "hue average [Â°]", "Leaf angle [Â°]", "Leaf area [mmÂ²]", "Leaf area index [mmÂ²/mmÂ²]", "Light penetration depth [mm]", "NDVI average"]
        
        X_raw = data_file[cols].values.astype(np.float32)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)

        self.data = torch.tensor(X_scaled, dtype=torch.float32)
        
        if "classification" in self.tasks:
            self.labels = torch.tensor(data_file['treatment'].values, dtype=torch.long)
        if "regression" in self.tasks:
            if reg_target == "hb":
                self.scores = torch.tensor(data_file['HB Score'].values, dtype=torch.long)
            elif reg_target == "pct":
                self.scores = torch.tensor(data_file['Avg Pct'].values, dtype=torch.long)
            
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def get_label(self, idx: int) -> tuple:
        output = []

        for t in self.tasks:
            if t == "classification":
                output.append(self.labels[idx])
            elif t == "regression":
                output.append(self.scores[idx])
        
        return tuple(output)
        
    def __getitem__(self, idx):
        x = self.data[idx]

        if self.transform:
            x = self.transform(x)

        output = [x]

        for t in self.tasks:
            if t == "classification":
                output.append(self.labels[idx])
            elif t == "regression":
                output.append(self.scores[idx])
        
        return tuple(output)
