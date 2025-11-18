# 文件: ddad_transformer.py
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from typing import Any
import torch
from torchvision import transforms
import os
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.font_manager import FontProperties

from unet import *
from visualize_steps import *
from reconstruction import Reconstruction
from metrics import *
from transformer_classifier import ClassificationTransformer

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
from scipy.ndimage import gaussian_filter
from torch.optim import Adam

from pathlib import Path

import importlib  # <--- (新) 添加这一行

# ... (所有 import 之后)

def _tensor_to_cv2_image(tensor_img: torch.Tensor) -> np.ndarray:
    """
    将 [C, H, W]、范围 [-1, 1] 的 PyTorch Tensor 转换为 [H, W, 3]、
    范围 [0, 255] 的 BGR (OpenCV) 图像。
    """
    img_np = tensor_img.cpu().numpy().transpose(1, 2, 0) # H, W, C
    # 反归一化 (假设你的 transform 是 Normalize(0.5, 0.5))
    img_np = (img_np * 0.5 + 0.5) * 255 
    img_np = img_np.astype(np.uint8)
    
    if img_np.shape[2] == 1:
        # 从灰度图转为 BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR) 
    elif img_np.shape[2] == 3:
        # 从 RGB (PyTorch/plt) 转为 BGR (OpenCV)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
        
    return img_np.copy()

def _cv2_image_to_plt(cv2_img_bgr: np.ndarray) -> np.ndarray:
    """将 [H, W, 3] 的 BGR (OpenCV) 图像转换为 RGB (Matplotlib) 图像。"""
    return cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)

# --- 可以添加在 ddad_transformer.py 的顶部 ---
class LocalizationSegmenter(nn.Module):
    """
    一个简单的全卷积网络 (FCN)，用于从聚合的 diff_map 预测异常掩码。
    它假设输入是 [B, 1, H, W] 的聚合特征图。
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(LocalizationSegmenter, self).__init__()
        # 编码器
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 瓶颈
        self.bottleneck = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 解码器
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        
        # 输出层
        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        
        # 瓶颈
        x_bottle = F.relu(self.bottleneck(self.pool(x2)))
        
        # 解码
        x_up1 = self.upconv1(x_bottle)
        x_up1 = F.relu(self.conv3(x_up1))
        
        x_up2 = self.upconv2(x_up1)
        
        # 输出
        # 我们使用 logits (原始值) 作为输出, 损失函数将处理 sigmoid
        logits = self.conv_out(x_up2) 
        return logits

# --- 放在 DDAD_Transformer_Analysis 同级别的位置 ---

class DiceBCELoss(nn.Module):
    """ 结合 Dice 损失和 BCE 损失，更稳定 """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs_logits, targets, smooth=1e-6):
        # inputs_logits 是模型的原始输出 (logits)
        inputs = torch.sigmoid(inputs_logits)       
        
        # --- BCE Loss ---
        bce_loss = F.binary_cross_entropy_with_logits(inputs_logits, targets, reduction='mean')
        
        # --- Dice Loss ---
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # 结合两种损失
        return bce_loss + dice_loss

class SupervisedLocalizationModule:
    def __init__(self, reconstruction_module: Reconstruction, config, device):
        self.reconstruction = reconstruction_module
        self.config = config
        self.device = device
        
        # 假设我们聚合 diff_maps 为 1 个通道
        self.segmenter = LocalizationSegmenter(in_channels=1, out_channels=1).to(device)
        
        self.optimizer = Adam(self.segmenter.parameters(), lr=1e-4)
        self.loss_fn = DiceBCELoss() # 使用 Dice+BCE 损失

        # 检查点路径
        clean_data_name = self.config.data.category.split("_")[0]
        self.checkpoint_dir = os.path.join(os.getcwd(), self.config.model.checkpoint_dir, clean_data_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.segmenter_checkpoint_path = "/data/xjy/DDAD/checkpoints_official/bottle/supervised_segmenter.pth"
    def _get_aggregated_map(self, input_data, w_localization=3, use_final_residual=False):
        """
        (复用逻辑) 从重建模块获取用于分割的特征图。
        """
        use_final_residual = True
        w_localization = 3
        self.reconstruction.unet.eval()
        with torch.no_grad():
            final_recon, _, _, diff_maps, _ = self.reconstruction(
                input_data, 
                input_data, 
                w_localization
            )
        
        if use_final_residual:
            # 策略1: 使用最终的残差
            residual_map = torch.abs(final_recon - input_data)
            aggregated_map = torch.mean(residual_map, dim=1, keepdim=True) # [B, 1, H, W]
        else:
            # 策略2: 使用 Diff maps 的均值
            abs_diff_maps = [torch.abs(d) for d in diff_maps]
            diff_stack = torch.stack(abs_diff_maps, dim=0) 
            aggregated_map_time = torch.mean(diff_stack, dim=0) # [B, C, H, W]
            aggregated_map = torch.mean(aggregated_map_time, dim=1, keepdim=True) # [B, 1, H, W]
            
        return aggregated_map

    def train(self, source_dataloader, num_epochs=20):
        """
        在源域上训练分割器。
        source_dataloader 必须返回 (images, masks, ...)
        """
        print(f"--- 🚀 开始有监督定位模型训练 (源域) ---")
        
        # 确保重建模型处于评估模式，我们只训练 segmenter
        self.reconstruction.unet.eval() 
        self.segmenter.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in source_dataloader:
                # 假设 dataloader 返回 (image, mask, label)
                # 您需要修改您的 Dataset_maker 来加载 mask
                images, masks, _ = batch 
                images = images.to(self.device)
                masks = masks.to(self.device) # [B, 1, H, W]

                # 1. 获取特征
                # 我们在训练时也使用 no_grad，因为我们不训练 UNet
                feature_map = self._get_aggregated_map(images)
                
                # 2. 预测掩码
                self.optimizer.zero_grad()
                pred_logits = self.segmenter(feature_map)
                
                # 3. 计算损失
                # 确保 mask 和 pred_logits 尺寸匹配
                # (如果需要，调整 mask 大小)
                if pred_logits.shape[2:] != masks.shape[2:]:
                    masks = F.interpolate(masks, size=pred_logits.shape[2:], mode='nearest')
                
                loss = self.loss_fn(pred_logits, masks)
                
                # 4. 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss / len(source_dataloader):.4f}")
        
        # 保存训练好的模型
        torch.save(self.segmenter.state_dict(), self.segmenter_checkpoint_path)
        print(f"--- ✅ 训练完成，模型已保存至: {self.segmenter_checkpoint_path} ---")

    def localize_on_target_domain(self, target_dataloader, num_samples=10, threshold=0.5):
        """
        在目标域上运行推理和可视化。
        target_dataloader 只需要返回 (images, ...)
        """
        print(f"--- 🚀 开始在目标域上进行异常定位 (推理) ---")
        
        # 加载训练好的模型
        if not os.path.exists(self.segmenter_checkpoint_path):
            print(f"错误: 找不到训练好的模型 {self.segmenter_checkpoint_path}")
            print("请先调用 .train() 方法在源域上进行训练。")
            return
            
        self.segmenter.load_state_dict(torch.load(self.segmenter_checkpoint_path, map_location=self.device))
        self.segmenter.eval()
        self.reconstruction.unet.eval()
        
        output_dir = 'supervised_localization_results'
        os.makedirs(output_dir, exist_ok=True)
        
        processed_samples = 0
        
        # (复用 DDAD_Transformer_Analysis 中的辅助函数)
        ddad_helper = self.reconstruction.unet # 借用一个实例来访问方法
        
        with torch.no_grad():
            for i, (input_data, _, labels) in enumerate(target_dataloader):
                if processed_samples >= num_samples:
                    break
                
                original_label = labels[0]
                print(f"正在处理目标域样本 {i+1} (类别: {original_label})...")
                
                input_data = input_data.to(self.device)
                
                # 1. 获取特征
                feature_map = self._get_aggregated_map(input_data)
                
                # 2. 预测掩码
                pred_logits = self.segmenter(feature_map)
                
                # 3. 后处理
                # 将预测结果调整回原始图像大小
                pred_logits_resized = F.interpolate(pred_logits, size=input_data.shape[2:], mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits_resized).squeeze(0) # [1, H, W]
                
                heatmap_norm = pred_prob.cpu().numpy().squeeze() # [H, W]
                binary_mask = (heatmap_norm > threshold).astype(np.uint8) * 255
                
                # 4. 可视化 (复用您的 `run_localization` 中的可视化逻辑)
                original_image_cv2 = _tensor_to_cv2_image(input_data[0])
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                output_image_cv2 = original_image_cv2.copy()
                cv2.drawContours(output_image_cv2, contours, -1, (0, 0, 255), 2) 
                
                heatmap_colored = cv2.applyColorMap(
                    (heatmap_norm * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                overlay_image = cv2.addWeighted(original_image_cv2, 0.6, heatmap_colored, 0.4, 0)

                # 5. 保存
                fig, axes = plt.subplots(1, 4, figsize=(20, 6))
                fig.suptitle(f"Target Sample {i+1} (Label: {original_label}) - Supervised Localization", fontsize=16)
                
                axes[0].imshow(_cv2_image_to_plt(original_image_cv2))
                axes[0].set_title("Original Target Image")
                axes[0].axis('off')

                axes[1].imshow(heatmap_norm, cmap='jet')
                axes[1].set_title(f"Predicted Heatmap (Supervised)")
                axes[1].axis('off')

                axes[2].imshow(_cv2_image_to_plt(overlay_image))
                axes[2].set_title("Heatmap Overlay")
                axes[2].axis('off')
                
                axes[3].imshow(_cv2_image_to_plt(output_image_cv2))
                axes[3].set_title(f"Circled (Thresh={threshold})")
                axes[3].axis('off')
                
                output_filename = os.path.join(output_dir, f"target_{self.config.data.category}_sample_{i+1}.png")
                plt.savefig(output_filename)
                plt.close(fig)
                print(f"✅ 目标域定位结果已保存至: {output_filename}")
                
                processed_samples += 1
                
        print(f"--- 🎉 目标域定位完成! 结果保存在 '{output_dir}' 文件夹 ---")

# --- 可视化和包装器代码 (保持不变) ---
def visualize_gradcam_for_transformer(cam, diff_steps, category, sample_index, is_average=False):
    """将一维的 CAM 结果可视化为条形图。"""
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(cam)), cam, color='skyblue')
    
    font_path = './simhei.ttf' 
    my_font = FontProperties(fname=font_path) if os.path.exists(font_path) else None

    title_prefix = "Average " if is_average else f"Sample {sample_index} "
    plt.xlabel('Denoising Step Index', fontproperties=my_font)
    plt.ylabel('Importance', fontproperties=my_font)
    plt.title(f'Grad-CAM for Transformer ({title_prefix}Category: {category})', fontproperties=my_font)
    
    tick_indices = np.linspace(0, len(diff_steps) - 1, num=min(len(diff_steps), 20), dtype=int)
    plt.xticks(ticks=tick_indices, labels=np.array(diff_steps)[tick_indices], rotation=45, fontsize=8)
    
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    output_dir = 'grad_cam_results'
    os.makedirs(output_dir, exist_ok=True)
    
    file_name_prefix = "average" if is_average else f'sample_{sample_index}'
    k = 0
    while os.path.exists(os.path.join(output_dir, f'{category}_{file_name_prefix}_gradcam_{k}.png')):
        k += 1
    
    output_path = os.path.join(output_dir, f'{category}_{file_name_prefix}_gradcam_{k}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Grad-CAM 可视化结果已保存至: {output_path}")

class TransformerCamWrapper(nn.Module):
    def __init__(self, model):
        super(TransformerCamWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 接收伪装的 "图像" 输入 [B, Feature_dim, Seq_len, 1]
        # 恢复成 Transformer 需要的序列形状 [B, Seq_len, Feature_dim]
        x_reshaped = x.squeeze(-1).permute(0, 2, 1)
        # 模型的输出是 (output, encoded_x)，包装器应该只返回 Grad-CAM 需要的 logits
        output, _ = self.model(x_reshaped)
        return output

class DDAD_Transformer_Analysis:
    def __init__(self, unet, config) -> None:
        self.unet = unet
        self.config = config
        import dataset
        
        # 2. (关键!) 强制 Python 重新加载该模块，
        #    清除由 'unet.py' 或其他 'import *' 引起的任何污染
        importlib.reload(dataset) 
        
        # 3. 现在我们可以安全地从这个干净的、重新加载的模块中访问 *类*
        self.test_dataset = dataset.Dataset_maker(
            root=config.data.data_dir, 
            category=config.data.category, 
            config=config, 
            is_train=False
        )
        self.reconstruction = Reconstruction(self.unet, self.config)
        
        input_dim = config.data.image_size * config.data.image_size * config.data.input_channel
        projection_dim = 512 
        seq_length = len(range(0, self.config.model.test_trajectoy_steps, self.config.model.skip))

        self.transformer = ClassificationTransformer(
            input_dim=input_dim, projection_dim=projection_dim, 
            num_heads=8, num_layers=2, num_classes=2, seq_length=seq_length
        ).to(self.config.model.device)

        self.optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=1e-4, weight_decay=1e-4)
        num_good = 0
        num_anomaly = 0
        # if hasattr(self.test_dataset, 'image_files'):
        #     for img_file_path in self.test_dataset.image_files:
        #         parent_dir_name = Path(img_file_path).parent.name
        #         if parent_dir_name == "Normal" or parent_dir_name == "good":
        #             num_good += 1
        #         else:
        #             # 假设其他所有文件夹（如 'Anomaly', 'scratch', 'crack'）都是异常
        #             num_anomaly += 1
        if hasattr(self.test_dataset, 'image_files'):
            # 注意：img_file_info 是一个元组 (path, label, mask)
            for img_file_info in self.test_dataset.image_files:
                # 只获取元组的第一个元素，即路径字符串
                img_path_str = img_file_info[0] 
                
                # 现在将路径字符串传递给 Path
                parent_dir_name = Path(img_path_str).parent.name 
                
                # 您的原始逻辑是基于文件夹名称区分好/坏样本。
                # 既然您在 Dataset_maker 中已经有了 'good' 或 'defective' 标签，
                # 更好的方法是直接使用标签来计算数量，避免依赖文件结构：
                
                label = img_file_info[1] # 获取标签字符串
                if label == 'good':
                    num_good += 1
                else:
                    num_anomaly += 1
        # bottle : 229, 63
        # carpet : 308, 89 
        print(f"DEBUG: num_good = {num_good}, num_anomaly = {num_anomaly}")
        total = num_good + num_anomaly
        weight_good, weight_anomaly = total / (2.0 * num_good), total / (2.0 * num_anomaly)
        class_weights = torch.tensor([weight_good, weight_anomaly], device=self.config.model.device)
        print(f"Using class weights: good={weight_good:.2f}, anomaly={weight_anomaly:.2f}")
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.num_epochs = 15
        clean_data_name = self.config.data.category.split("_")[0]
        self.checkpoint_dir = os.path.join(os.getcwd(), self.config.model.checkpoint_dir, clean_data_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # self.transformer_checkpoint_path = os.path.join(self.checkpoint_dir, 'transformer_cls_checkpoint.pth')
        # self.static_dataset_path = os.path.join(self.checkpoint_dir, 'transformer_static_ethat_dataset.pt')

        # 2. (新) diff_maps (像素差异) 特征的路径
        self.static_diff_maps_dataset_path = os.path.join(self.checkpoint_dir, 'transformer_static_diff_maps_dataset.pt')
        self.transformer_diff_maps_checkpoint_path = os.path.join(self.checkpoint_dir, 'transformer_cls_diff_maps_checkpoint.pth')
        # -----------------

    def _generate_and_save_static_ethat_dataset(self):
        """
        (原函数重命名)
        生成并保存 et_hat (预测噪声) 序列。
        注意: 这依赖于 self.reconstruction 返回 (final, et_hats)。
        """
        print(f"Generating static ET_HAT dataset for Transformer at {self.static_ethat_dataset_path}...")
        all_ethat_sequences, all_labels = [], []
        self.unet.eval()
        loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.config.model.num_workers)
        with torch.no_grad():
            for input_data, _, labels in loader:
                input_data = input_data.to(self.config.model.device)
                try:
                    # 假设你的 Reconstruction 返回值是 (final_reconstruction, et_hats)
                    _, et_hats = self.reconstruction(input_data, input_data, self.config.model.w)
                    ethat_flat = torch.stack(et_hats, dim=1).view(input_data.size(0), len(et_hats), -1)
                    all_ethat_sequences.append(ethat_flat.cpu())
                    all_labels.extend([0 if l == 'good' else 1 for l in labels])
                except Exception as e:
                    print(f"Error during et_hat generation (skipping batch): {e}")
                    print("This might be due to a mismatch in Reconstruction return values.")
                    continue

        if not all_ethat_sequences:
            print("No et_hat sequences were generated. Aborting.")
            return None, None
            
        all_ethats_tensor = torch.cat(all_ethat_sequences, dim=0)
        all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        torch.save({'ethat_sequences': all_ethats_tensor, 'labels': all_labels_tensor}, self.static_ethat_dataset_path)
        print("Static ET_HAT dataset saved successfully.")
        return all_ethats_tensor, all_labels_tensor

    # --- 你的新函数 ---
    def _generate_and_save_static_diff_maps_dataset(self):
        """
        (新函数)
        生成并保存 diff_maps (像素差异图) 序列。
        这依赖于 self.reconstruction 返回 (..., diff_maps, ...)。
        """
        print(f"Generating static DIFF_MAPS dataset for Transformer at {self.static_diff_maps_dataset_path}...")
        all_diff_map_sequences, all_labels = [], []
        self.unet.eval()
        loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.config.model.num_workers)
        with torch.no_grad():
            for input_data, _, labels in loader:
                input_data = input_data.to(self.config.model.device)
                try:
                    # 根据你提供的 Reconstruction.py:
                    # 返回: final, images_before, images_after, diff_maps, xs
                    _, _, _, diff_maps, _ = self.reconstruction(input_data, input_data, self.config.model.w)
                    
                    diff_map_flat = torch.stack(diff_maps, dim=1).view(input_data.size(0), len(diff_maps), -1)
                    all_diff_map_sequences.append(diff_map_flat.cpu())
                    all_labels.extend([0 if l == 'good' else 1 for l in labels])
                except Exception as e:
                    print(f"Error during diff_maps generation (skipping batch): {e}")
                    print("Ensure your Reconstruction class returns at least 4 values, with the 4th being diff_maps.")
                    continue
        
        if not all_diff_map_sequences:
            print("No diff_map sequences were generated. Aborting.")
            return None, None

        all_diff_maps_tensor = torch.cat(all_diff_map_sequences, dim=0)
        all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        torch.save({'diff_map_sequences': all_diff_maps_tensor, 'labels': all_labels_tensor}, self.static_diff_maps_dataset_path)
        print("Static DIFF_MAPS dataset saved successfully.")
        return all_diff_maps_tensor, all_labels_tensor

    def _train_transformer(self, static_data, static_labels, checkpoint_path):
        """ (修改) 训练函数现在接受一个 checkpoint 路径 """
        print(f"Starting CLS Transformer training, saving to {checkpoint_path}...")
        static_dataset = TensorDataset(static_data, static_labels)
        train_loader = DataLoader(static_dataset, batch_size=self.config.data.test_batch_size, shuffle=True)
        self.transformer.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for data_batch, labels_batch in train_loader:
                data_batch, labels_batch = data_batch.to(self.config.model.device), labels_batch.to(self.config.model.device)
                self.optimizer.zero_grad()
                output, _ = self.transformer(data_batch)
                loss = self.criterion(output, labels_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
        torch.save(self.transformer.state_dict(), checkpoint_path)

    def calculate_recall_at_k(self, labels, probabilities, k):
        # ... 此函数逻辑不变 ...
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_labels = labels[sorted_indices]
        top_k_labels = sorted_labels[:k]
        recall_at_k = np.sum(top_k_labels == 1) / k
        return recall_at_k

    def __call__(self, force_train=False, use_diff_maps=True) -> Any:
        """
        (修改) 主调用函数，增加 use_diff_maps 开关
        """
        
        # --- 1. 选择要使用的数据集路径 ---
        if use_diff_maps:
            print("--- Mode: Using DIFF_MAPS ---")
            static_dataset_path = self.static_diff_maps_dataset_path
            transformer_checkpoint_path = self.transformer_diff_maps_checkpoint_path
            generate_func = self._generate_and_save_static_diff_maps_dataset
            data_key = 'diff_map_sequences'
        else:
            print("--- Mode: Using ET_HATS ---")
            static_dataset_path = self.static_ethat_dataset_path
            transformer_checkpoint_path = self.transformer_ethat_checkpoint_path
            generate_func = self._generate_and_save_static_ethat_dataset
            data_key = 'ethat_sequences'

        # --- 2. 加载或生成数据集 ---
        if os.path.exists(static_dataset_path) and not force_train:
            print(f"Loading static dataset from file: {static_dataset_path}")
            dataset_dict = torch.load(static_dataset_path)
            static_data, static_labels = dataset_dict[data_key], dataset_dict['labels']
        else:
            static_data, static_labels = generate_func()
        
        if static_data is None:
            print("Failed to load or generate dataset. Exiting.")
            return

        # --- 3. 训练或加载 Transformer ---
        if force_train or not os.path.exists(transformer_checkpoint_path):
            self._train_transformer(static_data, static_labels, transformer_checkpoint_path)
        else:
            transformer_checkpoint_path = "/data/xjy/DDAD/checkpoints_DAGM/Class1/transformer_cls_diff_maps_checkpoint.pth"
            print(f"Loading pre-trained CLS transformer from {transformer_checkpoint_path}")
            self.transformer.load_state_dict(torch.load(transformer_checkpoint_path, map_location=self.config.model.device))
        
        # --- 4. 评估 ---
        print(f"Starting evaluation on {data_key} using CLS Transformer...")
        self.transformer.eval()
        all_predictions, all_probabilities = [], []
        static_eval_dataset = TensorDataset(static_data, static_labels)
        with torch.no_grad():
            for data_batch, _ in DataLoader(static_eval_dataset, batch_size=self.config.data.test_batch_size):
                data_batch = data_batch.to(self.config.model.device)
                preds, _ = self.transformer(data_batch)
                probabilities = torch.softmax(preds, dim=1).cpu().numpy()
                all_predictions.extend(torch.argmax(preds, dim=1).cpu().tolist())
                all_probabilities.extend(probabilities[:, 1])
        
        static_labels_np = static_labels.numpy().flatten()
        try:
            auroc = roc_auc_score(static_labels_np, all_probabilities)
            auprc = average_precision_score(static_labels_np, all_probabilities)
            num_anomalies = np.sum(static_labels_np == 1)
            if num_anomalies > 0:
                recall_at_k = self.calculate_recall_at_k(static_labels_np, np.array(all_probabilities), k=num_anomalies)
                print(f"Recall@{num_anomalies}: {recall_at_k:.4f}")
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
        except ValueError as e:
            print(f"Error in metric calculation: {e}")
    
    # --- (用这个版本替换) ---
    def run_localization(self, num_samples=5, ksize_blur=5, sigma_blur=1.5, threshold=0.5, use_final_residual=False):
        """
        (更新版) 
        - 增加了专门用于定位的 `w_localization` 参数。
        - (新) 增加了 "Reconstructed Image" 的可视化输出。
        """
        print(f"--- 🚀 开始执行异常定位 (Anomaly Localization) ---")
        
        # --- (新) 为定位设置专门的 w 值 ---
        # 尝试 0.5, 0.2, 0.0。
        # config.model.w (例如 4.0) 对于定位来说太高了！
        w_localization = 0.2 
        print(f"--- (重要) 使用专门的 w_localization: {w_localization} ---")
        
        if use_final_residual:
            method_name = f"Final Residual (w={w_localization})"
            print(f"--- 策略: {method_name}")
        else:
            method_name = f"Aggregated DiffMaps (w={w_localization})"
            print(f"--- 策略: {method_name}")
            
        print(f"将处理 {num_samples} 个样本...")
        
        loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, 
                            num_workers=self.config.model.num_workers)
        
        output_dir = 'localization_results'
        os.makedirs(output_dir, exist_ok=True)
        
        self.unet.eval()
        processed_samples = 0
        
        with torch.no_grad():
            for i, (input_data, _, labels) in enumerate(loader):
                if processed_samples >= num_samples:
                    break
                
                original_label = labels[0]
                print(f"正在处理样本 {i+1} (类别: {original_label})...")
                
                input_data = input_data.to(self.config.model.device)
                
                # 1. 运行重建 (使用我们新的 w_localization)
                final_recon, _, _, diff_maps, _ = self.reconstruction(
                    input_data, 
                    input_data, 
                    w_localization  # <--- (重要修改)
                )
                
                # 2. 根据策略选择热力图来源
                if use_final_residual:
                    residual_map = torch.abs(final_recon - input_data)
                    aggregated_map = residual_map.squeeze(0) 
                else:
                    abs_diff_maps = [torch.abs(d) for d in diff_maps]
                    diff_stack = torch.stack(abs_diff_maps, dim=0) 
                    aggregated_map = torch.mean(diff_stack, dim=0)
                    aggregated_map = aggregated_map.squeeze(0) 

                # ... (处理通道) ...
                if aggregated_map.shape[0] == 3: 
                    aggregated_map = torch.mean(aggregated_map, dim=0)
                else: 
                    aggregated_map = aggregated_map.squeeze(0) 
                heatmap_raw = aggregated_map.cpu().numpy()
                
                # 3. 后处理: 高斯模糊
                heatmap_smooth = gaussian_filter(heatmap_raw, sigma=sigma_blur) 

                # 4. 创建并应用对象掩码
                # ... (掩码逻辑保持不变) ...
                original_img_gray_np = input_data[0].cpu().numpy().transpose(1, 2, 0)
                if original_img_gray_np.shape[2] == 3:
                    original_img_gray_np = 0.299 * original_img_gray_np[:,:,0] + \
                                           0.587 * original_img_gray_np[:,:,1] + \
                                           0.114 * original_img_gray_np[:,:,2]
                else:
                    original_img_gray_np = original_img_gray_np.squeeze()
                original_img_gray_np = (original_img_gray_np * 0.5 + 0.5) * 255
                original_img_gray_np = original_img_gray_np.astype(np.uint8)
                _, object_mask = cv2.threshold(original_img_gray_np, 0, 255, 
                                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                heatmap_masked = heatmap_smooth * (object_mask / 255.0)
                
                # 5. 归一化和阈值化
                # ... (归一化逻辑保持不变) ...
                map_min, map_max = heatmap_masked.min(), heatmap_masked.max()
                heatmap_norm = (heatmap_masked - map_min) / (map_max - map_min + 1e-6)
                heatmap_norm = heatmap_norm * (object_mask / 255.0)
                binary_mask = (heatmap_norm > threshold).astype(np.uint8) * 255
                
                # 6. 可视化
                original_image_cv2 = _tensor_to_cv2_image(input_data[0])
                
                # --- (新) 将重建图也转为 CV2 格式 ---
                final_recon_cv2 = _tensor_to_cv2_image(final_recon[0])
                # ------------------------------------

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
                output_image_cv2 = original_image_cv2.copy()
                cv2.drawContours(output_image_cv2, contours, -1, (0, 0, 255), 2) 
                heatmap_colored = cv2.applyColorMap(
                    (heatmap_norm * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                overlay_image = original_image_cv2.copy()
                mask_indices = (object_mask == 255)
                overlay_image[mask_indices] = cv2.addWeighted(
                    original_image_cv2[mask_indices], 0.6, 
                    heatmap_colored[mask_indices], 0.4, 0
                )

                # 7. 保存绘图 (修改为 5 个子图)
                # --- (修改) 1x4 -> 1x5, 调整 figsize ---
                fig, axes = plt.subplots(1, 5, figsize=(25, 6)) 
                fig.suptitle(f"Sample {i+1} - Label: {original_label} - Category: {self.config.data.category}", 
                             fontsize=16)
                
                axes[0].imshow(_cv2_image_to_plt(original_image_cv2))
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                # --- (新) 添加重建图 ---
                axes[1].imshow(_cv2_image_to_plt(final_recon_cv2))
                axes[1].set_title(f"Reconstructed (w={w_localization})")
                axes[1].axis('off')
                
                # --- (修改) 索引 +1 ---
                axes[2].imshow(heatmap_norm, cmap='jet')
                axes[2].set_title(f"Heatmap ({method_name})") 
                axes[2].axis('off')
                
                # --- (修改) 索引 +1 ---
                axes[3].imshow(_cv2_image_to_plt(overlay_image))
                axes[3].set_title("Heatmap Overlay")
                axes[3].axis('off')
                
                # --- (修改) 索引 +1 ---
                axes[4].imshow(_cv2_image_to_plt(output_image_cv2))
                axes[4].set_title(f"Circled (Thresh={threshold})")
                axes[4].axis('off')
                
                output_filename = os.path.join(output_dir, 
                                               f"{self.config.data.category}_sample_{i+1}_{original_label}.png")
                plt.savefig(output_filename)
                plt.close(fig)
                print(f"✅ 定位结果已保存至: {output_filename}")
                
                processed_samples += 1
        
        print(f"--- 🎉 定位完成! 结果保存在 '{output_dir}' 文件夹 ---")
    # --- (将这个新方法添加到 DDAD_Transformer_Analysis 类中) ---

    def analyze_step_residuals(self, num_samples=5):
        """
        (新) 逐步残差 (Residual) 分析。
        
        这个方法在 DDAD_Transformer_Analysis 内部运行，
        因此可以正确访问已加载的 self.test_dataset 和 self.reconstruction。
        
        注意: 你的日志提到了 "Epsilon (et_hat)"，但你的代码
        (例如 run_localization, _generate_static_diff_maps_dataset)
        强烈表明重建模块返回的是 'diff_maps' (5个返回值)。
        
        此函数将遵循你的代码实现，分析 'diff_maps'。
        """
        print("--- 🚀 (已移入) 开始逐步 Residual (DiffMap) 分析 ---")
        
        # 1. 使用已经加载的数据集
        # self.test_dataset 已经在 __init__ 中被正确加载
        if not self.test_dataset:
            print("--- ❌ 错误: self.test_dataset 未初始化 ---")
            return
            
        print(f"正在使用已加载的数据集: {self.config.data.category}")
        
        # 2. 准备 Dataloader
        loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, 
                            num_workers=self.config.model.num_workers)
        
        # 3. 确保 unet 在评估模式
        self.unet.eval()
        
        print(f"将分析 {num_samples} 个样本...")
        
        output_dir = 'residual_analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        processed_samples = 0
        
        with torch.no_grad():
            for i, (input_data, _, labels) in enumerate(loader):
                if processed_samples >= num_samples:
                    break
                
                original_label = labels[0]
                print(f"--- 正在分析样本 {i+1} (类别: {original_label}) ---")
                
                input_data = input_data.to(self.config.model.device)
                
                try:
                    # 4. 运行重建以获取 diff_maps
                    # (这遵循了你 run_localization 中的 5 返回值结构)
                    final_recon, _, _, diff_maps, _ = self.reconstruction(
                        input_data, 
                        input_data, 
                        self.config.model.w # 使用配置中的 w
                    ) 
                    
                    print(f"  获取了 {len(diff_maps)} 个 diff_maps。")

                    # 5. (示例分析) 可视化第一个样本的 diff_maps 演变
                    if processed_samples < 3: # 为前 3 个样本绘制
                        print(f"  正在为样本 {i+1} 生成可视化图...")
                        
                        # (复用 run_localization 中的可视化辅助函数)
                        original_image_cv2 = _tensor_to_cv2_image(input_data[0])
                        final_recon_cv2 = _tensor_to_cv2_image(final_recon[0])
                        
                        # 图表布局: (原图, 重建图, diff_1, ..., diff_N, 聚合图)
                        num_plots = 2 + len(diff_maps) + 1
                        fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 3, 4))
                        if num_plots == 1: # 确保在 num_plots=1 时 axes 是可迭代的
                            axes = [axes]
                            
                        fig.suptitle(f"Step-wise Residual (DiffMap) 演变 - 样本 {i+1} ({original_label})")

                        axes[0].imshow(_cv2_image_to_plt(original_image_cv2))
                        axes[0].set_title("Original")
                        axes[0].axis('off')
                        
                        axes[1].imshow(_cv2_image_to_plt(final_recon_cv2))
                        axes[1].set_title(f"Final Recon (w={self.config.model.w})")
                        axes[1].axis('off')

                        all_diffs_np = []
                        for step_idx, diff_map in enumerate(diff_maps):
                            # (B, C, H, W) -> (H, W)
                            diff_map_viz = torch.mean(torch.abs(diff_map[0]), dim=0).cpu().numpy()
                            all_diffs_np.append(diff_map_viz)
                            
                            ax = axes[step_idx + 2]
                            im = ax.imshow(diff_map_viz, cmap='viridis')
                            ax.set_title(f"Diff Step {step_idx}")
                            ax.axis('off')
                        
                        # 聚合图
                        aggregated_map_np = np.mean(np.stack(all_diffs_np, axis=0), axis=0)
                        ax = axes[-1]
                        im = ax.imshow(aggregated_map_np, cmap='viridis')
                        ax.set_title("Aggregated DiffMap")
                        ax.axis('off')

                        output_filename = os.path.join(output_dir, f"diff_map_analysis_{self.config.data.category}_sample_{i+1}.png")
                        plt.savefig(output_filename)
                        plt.close(fig)
                        print(f"  ✅ 可视化结果已保存至: {output_filename}")

                    # 6. (示例分析) 计算 diff_maps 的 L2 范数
                    diff_map_norms = [torch.norm(d[0]) for d in diff_maps]
                    print(f"  diff_map 范数 (L2 Norms): {[f'{n:.2f}' for n in diff_map_norms]}")
                    
                    processed_samples += 1
                    
                except Exception as e:
                    print(f"--- ❌ 错误: 在分析样本 {i+1} 时出错 ---")
                    print(f"  原始错误: {e}")
                    print("  请确保 self.reconstruction() 返回 5 个值 (final, _, _, diff_maps, _)")
                    break # 停止循环

        print(f"--- 🎉 步骤分析完成! 结果保存在 '{output_dir}' 文件夹 ---")
        