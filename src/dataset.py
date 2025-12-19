import os
from typing import Optional, Callable, Dict, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .config import IMAGES_DIR
class GeoDataset(Dataset):
    #Dataset for country-level geolocation from images.
    def __init__(
        self,
        csv_path: str,
        images_dir: str = IMAGES_DIR,
        transform: Optional[Callable] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform

        if "image" not in self.df.columns or "country" not in self.df.columns:
            raise ValueError("CSV must contain 'image' and 'country' columns.")

        #Build label mapping: country name -> integer index
        self.countries: List[str] = sorted(self.df["country"].unique())
        self.label_to_idx: Dict[str, int] = {
            country: i for i, country in enumerate(self.countries)
        }
        self.idx_to_label: Dict[int, str] = {
            i: country for country, i in self.label_to_idx.items()
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row["image"]
        country = row["country"]

        img_path = os.path.join(self.images_dir, img_name)
        #Open image and ensure RGB
        image = Image.open(img_path).convert("RGB")
        #Convert country to numeric label
        label = self.label_to_idx[country]
        if self.transform:
            image = self.transform(image)
        return image, label
