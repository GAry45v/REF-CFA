import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter

from src.core.solver import DiffusionAnomalySolver
from src.data.factory import DataPipelineFactory
from src.utils.logger import get_root_logger
from src.modeling.architectures.segmentor import LocalizationSegmenter
from src.modeling.architectures.transformer import FeatureRefinementTransformer

class AnalysisEngine:
    """
    Integration of Transformer-based Discrimination, Supervised Localization, 
    and Step-wise Residual Analysis into the Ref-CFA framework.
    """
    def __init__(self, cfg, solver: DiffusionAnomalySolver = None):
        self.cfg = cfg
        self.logger = get_root_logger()
        self.device = torch.device(cfg.system.device_target)
        
        # Reuse the main solver components (UNet + Diffusion Engine)
        if solver is None:
            self.solver = DiffusionAnomalySolver(cfg)
            # Load checkpoint if specified in config or needed
            if cfg.solver.resume_checkpoint:
                self._load_solver_checkpoint(cfg.solver.resume_checkpoint)
        else:
            self.solver = solver
            
        self.solver.model.eval()
        
        # Paths
        clean_name = self.cfg.data_pipeline.test.category
        self.checkpoint_dir = os.path.join(self.cfg.system.output_root, "checkpoints", clean_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _load_solver_checkpoint(self, path):
        self.logger.info(f"Loading UNet backbone from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        # Handle DataParallel wrapping
        if 'module.' in list(checkpoint.keys())[0] and not isinstance(self.solver.model, nn.DataParallel):
             new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
             self.solver.model.load_state_dict(new_state_dict)
        else:
             self.solver.model.load_state_dict(checkpoint)

    def _get_diffusion_features(self, images, w_guidance=0.0):
        """
        Runs the diffusion reconstruction to get residuals/features.
        Adapts existing GaussianDiffusionEngine to return what we need.
        """
        # 1. Noise the image to T (or start from pure noise depending on strategy)
        # Here we assume reconstruction-based: x_0 -> x_T -> x_0'
        # For Ref-CFA, typically we noise to a certain step or use full trajectory.
        # Let's assume we use the engine's trajectory generation.
        
        t_map = torch.tensor([self.cfg.diffusion_process.params.steps - 1] * images.shape[0], device=self.device)
        noise = torch.randn_like(images)
        x_T = self.solver.diffusion_engine.q_sample(images, t_map, noise)
        
        # 2. Denoise and capture trajectory
        # Note: p_sample_loop_trajectory in your uploaded code returns {'diff_maps': [...]}
        ret = self.solver.diffusion_engine.p_sample_loop_trajectory(
            self.solver.model, 
            shape=images.shape, 
            noise=x_T, 
            w_guidance=w_guidance
        )
        return ret

    # ============================================================
    #  Feature 1: Transformer Based Analysis (DDAD_Transformer)
    # ============================================================
    
    def run_transformer_analysis(self, force_train=False):
        self.logger.info("--- Starting Transformer Analysis (DiffMaps) ---")
        
        # 1. Prepare Data
        static_path = os.path.join(self.checkpoint_dir, 'transformer_diff_maps.pt')
        if os.path.exists(static_path) and not force_train:
            self.logger.info(f"Loading static features from {static_path}")
            data_dict = torch.load(static_path)
            features, labels = data_dict['features'], data_dict['labels']
        else:
            features, labels = self._generate_static_dataset()
            torch.save({'features': features, 'labels': labels}, static_path)
            
        # 2. Setup Transformer
        # Calculate input dim based on image size (simple flattening assumption for now)
        # Or assuming diff_maps are pooled. Let's assume flattened diff_maps per step.
        # NOTE: In your original code, input_dim = H*W*C. This is huge. 
        # Ideally, we should pool before transformer. Keeping logic generic here.
        sample_feat = features[0, 0] # [Step, Dim]
        input_dim = sample_feat.shape[-1]
        seq_len = features.shape[1]
        
        transformer = FeatureRefinementTransformer(
            input_dim=input_dim, 
            seq_length=seq_len,
            num_classes=2
        ).to(self.device)
        
        ckpt_path = os.path.join(self.checkpoint_dir, 'transformer_ckpt.pth')
        
        # 3. Train or Load
        if force_train or not os.path.exists(ckpt_path):
            self._train_transformer(transformer, features, labels, ckpt_path)
        else:
            transformer.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            
        # 4. Evaluate
        self._evaluate_transformer(transformer, features, labels)

    def _generate_static_dataset(self):
        loader = DataPipelineFactory.create_dataloader(
            self.cfg.data_pipeline.test, batch_size=1, num_workers=4, is_train=False
        )
        all_feats, all_labels = [], []
        
        self.logger.info("Generating static features for Transformer...")
        with torch.no_grad():
            for img, _, label in loader:
                img = img.to(self.device)
                ret = self._get_diffusion_features(img)
                diff_maps = ret['diff_maps'] # List of [B, C, H, W]
                
                # Process diff_maps to vectors [B, Steps, Dim]
                # Flattening H*W*C
                seq = [d.flatten(1) for d in diff_maps] 
                seq = torch.stack(seq, dim=1) # [B, Steps, Dim]
                
                all_feats.append(seq.cpu())
                # Ensure label is int
                lbl = 1 if (isinstance(label, torch.Tensor) and label.item() == 1) or label == 1 else 0
                all_labels.append(lbl)
                
        return torch.cat(all_feats, dim=0), torch.tensor(all_labels)

    def _train_transformer(self, model, features, labels, save_path):
        self.logger.info("Training Transformer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        model.train()
        
        for epoch in range(10): # Configurable epochs
            total_loss = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.logger.info(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
        
        torch.save(model.state_dict(), save_path)

    def _evaluate_transformer(self, model, features, labels):
        model.eval()
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=16)
        probs, true_labels = [], []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                logits, _ = model(x)
                prob = torch.softmax(logits, dim=1)[:, 1]
                probs.extend(prob.cpu().numpy())
                true_labels.extend(y.numpy())
                
        auroc = roc_auc_score(true_labels, probs)
        self.logger.info(f"Transformer Evaluation Results - AUROC: {auroc:.4f}")

    # ============================================================
    #  Feature 2: Supervised Localization (Target Domain Adaptation)
    # ============================================================

    def train_supervised_localization(self, num_epochs=20):
        self.logger.info("--- Starting Supervised Localization Training (Source Domain) ---")
        
        # 1. Setup Data (Use Source Config but filter for anomalies)
        # Note: You need to ensure the config points to the source dataset
        # Mocking a filter for anomalies here, assuming pipeline supports it
        train_loader = DataPipelineFactory.create_dataloader(
             self.cfg.data_pipeline.train, batch_size=8, num_workers=4, is_train=True
        ) 
        
        # 2. Setup Segmenter
        segmenter = LocalizationSegmenter().to(self.device)
        optimizer = torch.optim.Adam(segmenter.parameters(), lr=1e-4)
        # Simple BCE for brevity, can use DiceBCELoss from your code
        criterion = nn.BCEWithLogitsLoss() 
        
        segmenter.train()
        self.solver.model.eval()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            steps = 0
            for batch in train_loader:
                # Assuming loader returns (img, label) or (img, mask, label)
                # We need masks for supervised training
                if len(batch) == 3:
                    img, mask, label = batch
                else:
                    continue # Skip if no mask
                
                # Only train on anomalies
                if label.sum() == 0: continue

                img = img.to(self.device)
                mask = mask.to(self.device)
                
                with torch.no_grad():
                    # Get aggregated map from diffusion
                    ret = self._get_diffusion_features(img)
                    diff_maps = ret['diff_maps']
                    # Aggregation strategy: Mean of Abs Diffs
                    diff_stack = torch.stack([d.abs() for d in diff_maps], dim=1) # [B, T, C, H, W]
                    feat_map = diff_stack.mean(dim=1).mean(dim=1, keepdim=True) # [B, 1, H, W]
                
                optimizer.zero_grad()
                preds = segmenter(feat_map)
                
                # Resize if needed
                if preds.shape != mask.shape:
                    mask = F.interpolate(mask, size=preds.shape[2:], mode='nearest')
                    
                loss = criterion(preds, mask)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            
            if steps > 0:
                self.logger.info(f"Epoch {epoch+1} Loss: {epoch_loss/steps:.4f}")
                
        save_path = os.path.join(self.checkpoint_dir, 'supervised_segmenter.pth')
        torch.save(segmenter.state_dict(), save_path)
        self.logger.info(f"Saved segmenter to {save_path}")

    def run_supervised_inference(self, num_samples=5):
        self.logger.info("--- Running Supervised Localization Inference (Target Domain) ---")
        
        ckpt_path = os.path.join(self.checkpoint_dir, 'supervised_segmenter.pth')
        if not os.path.exists(ckpt_path):
            self.logger.error("No supervised segmenter checkpoint found!")
            return
            
        segmenter = LocalizationSegmenter().to(self.device)
        segmenter.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        segmenter.eval()
        
        test_loader = DataPipelineFactory.create_dataloader(
            self.cfg.data_pipeline.test, batch_size=1, num_workers=1, is_train=False
        )
        
        count = 0
        output_dir = os.path.join(self.cfg.system.output_root, "vis_supervised")
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for img, _, label in test_loader:
                if count >= num_samples: break
                img = img.to(self.device)
                
                # Extract Feature
                ret = self._get_diffusion_features(img)
                diff_stack = torch.stack([d.abs() for d in ret['diff_maps']], dim=1)
                feat_map = diff_stack.mean(dim=1).mean(dim=1, keepdim=True)
                
                # Predict
                logits = segmenter(feat_map)
                prob = torch.sigmoid(logits)
                
                # Viz
                self._visualize_segmentation(img, prob, count, output_dir)
                count += 1

    def _visualize_segmentation(self, img, pred_mask, idx, out_dir):
        # Simple viz helper
        img_np = (img[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5) * 255
        mask_np = pred_mask[0,0].cpu().numpy() * 255
        mask_heatmap = cv2.applyColorMap(mask_np.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np.astype(np.uint8), 0.6, mask_heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(out_dir, f"sup_loc_{idx}.png"), overlay)

    # ============================================================
    #  Feature 3: Step-wise Residual Analysis
    # ============================================================
    
    def analyze_residuals(self, num_samples=3):
        self.logger.info("--- Running Step-wise Residual Analysis ---")
        loader = DataPipelineFactory.create_dataloader(
            self.cfg.data_pipeline.test, batch_size=1, num_workers=1, is_train=False
        )
        
        output_dir = os.path.join(self.cfg.system.output_root, "vis_residuals")
        os.makedirs(output_dir, exist_ok=True)
        
        count = 0
        with torch.no_grad():
            for img, _, _ in loader:
                if count >= num_samples: break
                img = img.to(self.device)
                
                ret = self._get_diffusion_features(img)
                diff_maps = ret['diff_maps']
                
                # Visualize evolution
                fig, axes = plt.subplots(1, min(5, len(diff_maps)), figsize=(15, 3))
                indices = np.linspace(0, len(diff_maps)-1, len(axes), dtype=int)
                
                for i, ax_idx in enumerate(indices):
                    diff = diff_maps[ax_idx][0].mean(0).cpu().numpy() # H, W
                    axes[i].imshow(diff, cmap='viridis')
                    axes[i].set_title(f"Step {ax_idx}")
                    axes[i].axis('off')
                    
                plt.savefig(os.path.join(output_dir, f"residual_step_{count}.png"))
                plt.close()
                count += 1