from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import PIL


# ------------ Costum Dataset Class ------------
class ThreeLinesScatterPlot(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(len(self.img_labels))
  
    def __getitem__(self, idx):
        split_sub_dir = self.img_labels.iloc[idx,0].split('_')
        sub_dir = split_sub_dir[0]
        img_path = os.path.join(self.img_dir, sub_dir, self.img_labels.iloc[idx, 0]) # image path

        image = np.array(PIL.Image.open(img_path).convert('RGB'), dtype=np.float32)
        label = self.img_labels.iloc[idx,1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
