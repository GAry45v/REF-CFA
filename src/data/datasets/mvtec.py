import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.core.registry import DATASET_REGISTRY
from src.data.transform import TransformFactory

@DATASET_REGISTRY.register("MVTecDataset")
class MVTecDataset(Dataset):
    """
    A standardized interface for the MVTec Anomaly Detection Dataset.
    """
    def __init__(self, root: str, category: str, is_train: bool = True, config: dict = None):
        self.root = root
        self.category = category
        self.is_train = is_train
        self.config = config or {}
        
        # Utilize the TransformFactory for image preprocessing
        image_size = self.config.get('image_size', 256)
        self.transform = TransformFactory.create_standard_transforms(image_size)
        self.mask_transform = TransformFactory.create_mask_transforms(image_size)

        self.image_files = self._discover_files()

    def _discover_files(self):
        if self.is_train:
            # Pattern: root/category/train/good/*.png
            path_pattern = os.path.join(self.root, self.category, "train", "good", "*.png")
        else:
            # Pattern: root/category/test/**/*/*.png
            path_pattern = os.path.join(self.root, self.category, "test", "*", "*.png")
        
        files = sorted(glob.glob(path_pattern))
        if not files:
            raise RuntimeError(f"No images found at {path_pattern}. Check your data root.")
        return files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        
        # Load Image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Determine Label
        parent_dir = os.path.dirname(image_path)
        is_good = parent_dir.endswith("good")
        label = 0 if is_good else 1  # 0 for Normal, 1 for Anomaly

        if self.is_train:
            return image, label
        else:
            # Test mode logic: Load Ground Truth Mask
            target = torch.zeros((1, image.shape[1], image.shape[2]))
            if not is_good:
                # Construct mask path logic
                # MVTec assumption: .../test/crack/000.png -> .../ground_truth/crack/000_mask.png
                mask_path = image_path.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
                if os.path.exists(mask_path):
                    mask_img = Image.open(mask_path).convert("L")
                    target = self.mask_transform(mask_img)
            
            return image, target, label