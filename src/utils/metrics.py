import torch
from torchmetrics import AUROC, AveragePrecision
from typing import Dict
import numpy as np

class AnomalyMetricEvaluator:
    """
    Encapsulates the logic for computing image-level and pixel-level anomaly detection metrics.
    """
    def __init__(self):
        self.preds_image = []
        self.labels_image = []
        self.preds_pixel = []
        self.masks_pixel = []

    def update(self, pred_score: float, label: int):
        """Update image-level metrics"""
        self.preds_image.append(pred_score)
        self.labels_image.append(label)

    def compute(self) -> Dict[str, float]:
        """
        Computes the final AUROC and AUPRC scores.
        """
        preds = torch.tensor(self.preds_image)
        labels = torch.tensor(self.labels_image)
        
        auroc_fn = AUROC(task="binary")
        auprc_fn = AveragePrecision(task="binary")
        
        return {
            "Image_AUROC": auroc_fn(preds, labels).item(),
            "Image_AUPRC": auprc_fn(preds, labels).item()
        }