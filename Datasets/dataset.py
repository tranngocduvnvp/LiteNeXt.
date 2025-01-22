import torch.utils.data as data
import numpy as np
from PIL import Image

class MedicalDataset(data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, state=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask[mask >= 1] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask.unsqueeze(0).float()

