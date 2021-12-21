from torch.utils.data import Dataset
from helper_functions import get_sub_dir
from helper_functions import get_image
import PIL
import os
import numpy as np
import pandas as pd


# ------------ Costum Dataset Class ------------
class PhytoplanktonImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform, num_classes, percent, dims, augs):
        self.img_labels = pd.read_csv(annotations_file)  # Image name and label file loaded into img_labels
        self.img_dir = img_dir  # directory to find all image names
        self.transform = transform  # tranforms to apply to images
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.percent = percent
        self.dims = dims
        self.augs = augs

    def __len__(self):
        return int(len(self.img_labels) * (self.percent / 100))
  

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, get_sub_dir(self.img_labels.iloc[idx, 0]),self.img_labels.iloc[idx, 0]) # image path
        image = get_image(np.array(PIL.Image.open(img_path).convert('RGB'), dtype=np.uint8), self.augs)
        #image = np.array(PIL.Image.open(img_path).convert('RGB'), dtype=np.float32)
        # no longer using .convert('RGB')
        classification = self.img_labels.iloc[idx,1] # getting label from csv
        label = (classification - 1)
        if self.num_classes == 2:
            if classification == 65: # Detritus vs. non detritus
                label = 0
            else:
                label = 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
