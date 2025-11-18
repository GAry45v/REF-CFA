import os
import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.core.registry import DATASET_REGISTRY
from src.data.transform import TransformFactory

@DATASET_REGISTRY.register("VisADataset")
class VisADataset(Dataset):
    """
    Dataset adapter for the VisA (Visual Anomaly) benchmark.
    Supports rigorous split protocols and mask loading.
    """
    def __init__(self, root: str, category: str, is_train: bool = True, config: dict = None):
        self.root = Path(root)
        self.category = category
        self.is_train = is_train
        self.config = config or {}
        
        image_size = self.config.get('image_size', 256)
        self.transform = TransformFactory.create_standard_transforms(image_size)
        self.mask_transform = TransformFactory.create_mask_transforms(image_size)
        
        # Complex path logic specific to VisA structure
        # Root/category/Data/Images/[Normal|Anomaly]
        self.image_base_path = self.root / self.category / "Data" / "Images"
        self.mask_base_path = self.root / self.category / "Data" / "Masks"
        
        self.samples = self._discover_samples()
        
        print(f"[VisADataset] Loaded {len(self.samples)} images for category '{category}' (Train={is_train})")

    def _discover_samples(self):
        samples = []
        
        # VisA file extensions
        extensions = ["*.jpg", "*.JPG", "*.png"]
        
        def _glob_images(folder):
            files = []
            for ext in extensions:
                files.extend(list(folder.glob(ext)))
            return sorted(files)

        normal_dir = self.image_base_path / "Normal"
        anomaly_dir = self.image_base_path / "Anomaly"

        if self.is_train:
            # Standard Protocol: Train on Normal samples only
            # (Unless config overrides for supervised training)
            if self.config.get("supervised", False):
                samples.extend([(p, 0) for p in _glob_images(normal_dir)])
                samples.extend([(p, 1) for p in _glob_images(anomaly_dir)])
            else:
                samples.extend([(p, 0) for p in _glob_images(normal_dir)])
        else:
            # Test: Normal + Anomaly
            samples.extend([(p, 0) for p in _glob_images(normal_dir)])
            samples.extend([(p, 1) for p in _glob_images(anomaly_dir)])
            
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Load Mask (if anomaly)
        if label == 1:
            # VisA mask logic: Filename corresponds to image stem
            mask_path = self.mask_base_path / "Anomaly" / f"{img_path.stem}.png"
            if not mask_path.exists():
                 # Fallback: try root mask path
                 mask_path = self.mask_base_path / f"{img_path.stem}.png"
            
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                target = self.mask_transform(mask)
            else:
                # Fallback if mask missing
                target = torch.zeros((1, image.shape[1], image.shape[2]))
        else:
            target = torch.zeros((1, image.shape[1], image.shape[2]))

        return image, target, label