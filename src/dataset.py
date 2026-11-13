import os 
import cv2 
import numpy as np
import torch 
from torch.utils.data import Dataset

def mask_name_for(image_name):
    """cxrimage_1012.png -> cxrmask_1012.jpeg"""
    number = image_name.replace("cxrimage_", "").split(".")[0]
    return f"cxrmask_{number}.jpeg"


class LungDataset(Dataset):
    def __init__(self, split_dir, img_size=256):
        self.image_dir = os.path.join(split_dir, "image")
        self.mask_dir = os.path.join(split_dir, "mask")
        self.img_size = (img_size, img_size)
        self.filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        image_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, mask_name_for(name))

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)  # threshold: JPEG masks are noisy

        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)
        return image, mask