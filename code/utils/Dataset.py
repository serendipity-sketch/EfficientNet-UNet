from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, num_classes=2):
        self.image_paths = list_images(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.num_classes = num_classes

        if not self.image_paths:
            raise ValueError(f"No supported images found in: {image_dir}")
        missing = [path.name for path in self.image_paths if not (self.label_dir / path.name).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing {len(missing)} labels in {self.label_dir}: {preview}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(self.label_dir / image_path.name).convert("L")).copy()

        # The source masks use 255 for foreground. Keep 255 as an ignore label
        # only when it coexists with already encoded classes beyond binary data.
        if self.num_classes == 2:
            mask = (mask > 0).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return {"img": image, "label": mask.long(), "name": image_path.name}


class PredictionDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = list_images(image_dir)
        self.transform = transform
        if not self.image_paths:
            raise ValueError(f"No supported images found in: {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        pil_image = Image.open(path).convert("RGB")
        original_size = pil_image.size  # width, height
        image = self.transform(image=np.asarray(pil_image))["image"]
        return {"img": image, "name": path.name, "original_size": original_size}
