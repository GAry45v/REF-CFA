from torch.utils.data import DataLoader, Subset
from src.core.registry import DATASET_REGISTRY
from typing import Optional, List
import os

# 假设原来的 dataset.py 被重构为 src/data/datasets/mvtec.py 并注册为 "MVTecDataset"
from src.data.datasets.mvtec import MVTecDataset 

class DataPipelineFactory:
    """
    A factory class responsible for constructing the data ingestion pipeline.
    Handles dataset instantiation, transform composition, and dataloader configuration.
    """
    
    @staticmethod
    def create_dataloader(
        dataset_cfg: dict,
        batch_size: int,
        num_workers: int,
        is_train: bool = True,
        shuffle: bool = True,
        drop_last: bool = True
    ) -> DataLoader:
        
        dataset_type = dataset_cfg.get("type", "MVTecDataset")
        
        # Log the creation process
        print(f"[DataFactory] Initializing dataset: {dataset_type} | Mode: {'Train' if is_train else 'Test'}")
        
        # Build dataset via Registry (Mocking the call to the actual registered class)
        # In a real run, MVTecDataset would be pulled from DATASET_REGISTRY
        dataset = MVTecDataset(
            root=dataset_cfg['root'],
            category=dataset_cfg['category'],
            config=dataset_cfg.get('extra_config', None), # Pass the full config object if needed
            is_train=is_train
        )
        
        # Complex Subset logic (e.g., for Few-Shot or debugging)
        if dataset_cfg.get("use_subset", False):
            indices = list(range(dataset_cfg.get("subset_size", 100)))
            print(f"[DataFactory] Reducing dataset to subset of size {len(indices)}.")
            dataset = Subset(dataset, indices)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True
        )
        
        return loader