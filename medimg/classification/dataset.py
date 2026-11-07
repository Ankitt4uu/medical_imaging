import csv

import torch
from PIL import Image
from torch.utils.data import Dataset

from medimg.config import CLASSES


class CXRDataset(Dataset):
    def __init__(self, csv_path, transform):
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        y = torch.tensor([float(r[c]) for c in CLASSES])
        return self.transform(img), y
