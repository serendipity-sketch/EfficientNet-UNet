import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, paths, transform=None, class_num=1):
        self.img_dir = paths[0]
        self.label_dir = paths[1]
        self.transform = transform
        self.class_num = class_num

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        img_np = np.array(img)
        mask_np = np.array(label)
        mask_np[mask_np == 255] = 1

        if self.transform is not None:
            aug = self.transform(image=img_np, mask=mask_np)
            img = aug['image']
            label = aug['mask']

        label = label.long()

        return {"img": img, "label": label}