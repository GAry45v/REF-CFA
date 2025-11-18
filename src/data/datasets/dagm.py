import os
import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.core.registry import DATASET_REGISTRY
from src.data.transform import TransformFactory

@DATASET_REGISTRY.register("DAGMDataset")
class DAGMDataset(Dataset):
    """
    Dataset adapter for the DAGM 2007 dataset.
    Handles the specific folder structure (ClassX / ClassX_def) and mask naming conventions.
    """
    def __init__(self, root: str, category: str, is_train: bool = True, config: dict = None):
        self.root = Path(root)
        self.category = category # e.g., "Class1"
        self.is_train = is_train
        self.config = config or {}
        
        image_size = self.config.get('image_size', 256)
        self.transform = TransformFactory.create_standard_transforms(image_size)
        self.mask_transform = TransformFactory.create_mask_transforms(image_size)
        
        self.samples = self._make_dataset()
        print(f"[DAGMDataset] Loaded {len(self.samples)} images for {category}.")

    def _make_dataset(self):
        samples = []
        
        # Normal folder: root/Class1/
        normal_dir = self.root / self.category
        # Defective folder: root/Class1_def/
        defective_dir = self.root / f"{self.category}_def"
        
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        # Helper to check extension
        is_img = lambda p: p.suffix.lower() in valid_exts and "_mask" not in p.name

        if self.is_train:
            # Train on Normal only
            for p in normal_dir.iterdir():
                if is_img(p): samples.append((str(p), 0, None))
        else:
            # Test on Normal + Defective
            for p in normal_dir.iterdir():
                if is_img(p): samples.append((str(p), 0, None))
            
            # Pre-scan masks for defective images
            # DAGM convention: image.png -> image_mask.png
            mask_map = {}
            for p in defective_dir.iterdir():
                if "_mask" in p.name:
                    # Map "0001_mask.png" -> "0001"
                    key = p.stem.replace("_mask", "")
                    mask_map[key] = str(p)

            for p in defective_dir.iterdir():
                if is_img(p):
                    mask_path = mask_map.get(p.stem, None)
                    samples.append((str(p), 1, mask_path))
                    
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, mask_path = self.samples[index]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        if label == 1 and mask_path:
            mask = Image.open(mask_path).convert("L")
            target = self.mask_transform(mask)
        else:
            target = torch.zeros((1, image.shape[1], image.shape[2]))
            
        return image, target, label