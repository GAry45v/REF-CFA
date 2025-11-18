import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from typing import List, Optional, Union
from torchvision import transforms
from pathlib import Path

class Visualizer:
    """
    A comprehensive visualization engine for diffusion-based anomaly detection.
    Handles tensor denormalization, heatmap generation, and trajectory plotting.
    """
    def __init__(self, output_dir: str, mean: List[float] = [0.5, 0.5, 0.5], std: List[float] = [0.5, 0.5, 0.5]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def _denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalized (N, C, H, W) or (C, H, W) tensor to a (H, W, C) numpy image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * self.std + self.mean) * 255
        return np.clip(img, 0, 255).astype(np.uint8)

    def plot_diffusion_trajectory(self, trajectory: List[torch.Tensor], sample_id: str, select_steps: int = 10):
        """
        Visualizes the reverse diffusion process x_T -> x_0.
        """
        num_frames = len(trajectory)
        indices = np.linspace(0, num_frames - 1, select_steps, dtype=int)
        
        fig, axes = plt.subplots(1, select_steps, figsize=(select_steps * 2, 3))
        
        for i, idx in enumerate(indices):
            img = self._denormalize(trajectory[idx])
            axes[i].imshow(img)
            axes[i].set_title(f"t={num_frames - 1 - idx}", fontsize=8)
            axes[i].axis('off')
            
        save_path = self.output_dir / f"trajectory_{sample_id}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def save_anomaly_map(self, image: torch.Tensor, anomaly_map: torch.Tensor, sample_id: str, threshold: float = 0.5):
        """
        Overlays the anomaly heatmap on the original image and saves the result.
        """
        # Prepare Original Image
        orig_img = self._denormalize(image) # H, W, 3 (RGB)
        orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        # Prepare Heatmap
        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.squeeze().cpu().numpy()
        
        # Normalize heatmap to 0-255
        heatmap_norm = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Blend
        overlay = cv2.addWeighted(orig_img_bgr, 0.6, heatmap_color, 0.4, 0)
        
        # Contours for segmentation
        binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # Save
        cv2.imwrite(str(self.output_dir / f"anomaly_{sample_id}.png"), overlay)

    def visualize_residuals(self, diff_maps: List[torch.Tensor], sample_id: str):
        """
        Visualizes the step-wise residuals (epsilon_hat or x_prev - x_t).
        """
        # Aggregate residuals for visualization (e.g., L2 norm across channels)
        # Select a few key steps
        indices = np.linspace(0, len(diff_maps) - 1, 5, dtype=int)
        
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        
        # Plot total accumulated residual
        total_residual = torch.stack(diff_maps).sum(dim=0).abs().mean(dim=1).squeeze().cpu().numpy()
        axes[0].imshow(total_residual, cmap='inferno')
        axes[0].set_title("Total Residual")
        axes[0].axis('off')

        for i, idx in enumerate(indices):
            # Calculate magnitude of difference at this step
            diff = diff_maps[idx].abs().mean(dim=1).squeeze().cpu().numpy() # B, C, H, W -> H, W
            axes[i+1].imshow(diff, cmap='viridis')
            axes[i+1].set_title(f"Step {idx} Diff")
            axes[i+1].axis('off')

        plt.savefig(self.output_dir / f"residuals_{sample_id}.png")
        plt.close(fig)