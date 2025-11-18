# # 文件: visualize_steps.py
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from torchvision import transforms
# from typing import List

# # (show_tensor_image 和 visualize_step_analysis_simple 函数保持不变)
# def show_tensor_image(image):
#     # (此函数保持不变)
#     if len(image.shape) == 4: image = image[0, :, :, :]
#     reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: (t + 1) / 2 if t.min() < 0 else t), # 兼容 [0,1] 和 [-1,1]
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)),
#         transforms.Lambda(lambda t: t.clamp(0, 1)), # 确保值在[0,1]
#         transforms.Lambda(lambda t: t.cpu().numpy()),
#     ])
#     return reverse_transforms(image)

# def visualize_step_analysis_simple(step_norms, category):
#     # (此函数保持不变, 用于绘制范数曲线)
#     num_total_steps = len(step_norms)
#     # --- 修正：argsort 默认升序，取最后几个才是最大的 ---
#     key_step_indices = np.argsort(step_norms)[-5:] # 取最大的5个
#     plt.figure(figsize=(15, 6))
#     plt.plot(range(num_total_steps), step_norms, marker='.', linestyle='-', label='Feature Norm')
#     # --- 修正：标签应该是 Top 5 ---
#     # plt.scatter(key_step_indices, np.array(step_norms)[key_step_indices], color='red', s=120, zorder=5, label='Top 5 Key Steps')
#     plt.title(f'L2 Norm of Feature Vector at Each Denoising Step (Category: {category})', fontsize=16)
#     plt.xlabel('Denoising Step Index')
#     plt.ylabel('L2 Norm (Importance)')
#     plt.grid(True)
#     plt.legend()
#     plt.xticks(np.arange(0, num_total_steps, max(1, num_total_steps // 20))) # 优化x轴刻度显示
#     if not os.path.exists('transfer_learning'): os.mkdir('transfer_learning')
#     k = 0
#     while os.path.exists(f'transfer_learning/{category}_simple_step_analysis_{k}.png'): k += 1
#     plt.savefig(f'transfer_learning/{category}_simple_step_analysis_{k}.png', bbox_inches='tight')
#     plt.close()
#     print(f"Simple step analysis visualization saved to 'transfer_learning/{category}_simple_step_analysis_{k}.png'")


# # --- 新增 v2 版本：更直观的详细关键步骤可视化函数 ---
# def visualize_key_step_details_v2(key_step_indices, images_before, images_after, diff_maps, category):
#     """
#     为每个关键步骤，并排显示：去噪前、去噪后、变化幅度图、热力图叠加。
#     V2版本改进了差异图的计算和可视化方式，使其更直观。
#     """
#     num_key_steps = len(key_step_indices)
#     fig, axes = plt.subplots(num_key_steps, 4, figsize=(20, 5 * num_key_steps))
    
#     if num_key_steps == 1:
#         axes = np.expand_dims(axes, axis=0)
        
#     for i, step_idx in enumerate(sorted(key_step_indices)): # 建议按步骤顺序可视化
#         # 提取对应步骤的数据
#         before_img = images_before[step_idx]
#         after_img = images_after[step_idx]
#         diff_img = diff_maps[step_idx]

#         # --- 核心改进 1: 计算变化的幅度 (L2范数) ---
#         # diff_img 的 shape 是 (C, H, W)，我们需要在 Channel 维度上计算范数
#         # 这能准确捕捉RGB颜色的总体变化大小，而不是简单平均
#         change_magnitude = torch.linalg.norm(diff_img.float(), ord=2, dim=0)
        
#         # 归一化到 [0, 1] 以便显示
#         if change_magnitude.max() > 0:
#             change_magnitude_norm = change_magnitude / change_magnitude.max()
#         else:
#             change_magnitude_norm = change_magnitude # 如果没有变化，则为全0

#         # 1. 去噪前
#         axes[i, 0].imshow(show_tensor_image(before_img))
#         axes[i, 0].set_title(f"Step {step_idx}: Before Denoising")
#         axes[i, 0].axis('off')

#         # 2. 去噪后
#         after_img_np = show_tensor_image(after_img)
#         axes[i, 1].imshow(after_img_np)
#         axes[i, 1].set_title(f"Step {step_idx}: After Denoising")
#         axes[i, 1].axis('off')

#         top_k_percent = 5
#         # 4. 生成二值掩码图 (Mask)
#         if change_magnitude_norm.numel() > 0 and change_magnitude_norm.max() > 0:
#             # 计算阈值，只取变化最剧烈的前 top_k_percent%
#             threshold = torch.quantile(change_magnitude_norm.flatten(), 1 - (top_k_percent / 100.0))
#             mask = (change_magnitude_norm >= threshold).cpu().numpy()
#         else:
#             # 如果没有变化，掩码为空
#             mask = np.zeros_like(change_magnitude_norm.cpu().numpy(), dtype=bool)
#         # 3. 变化幅度 (灰度图) - 更准确的“Difference”图
#         axes[i, 2].imshow(mask, cmap='gray')
#         axes[i, 2].set_title(f"Step {step_idx}: Change Magnitude")
#         axes[i, 2].axis('off')
        
#         # --- 核心改进 2: 将热力图叠加在“去噪后”的图像上 ---
#         # 4. 热力图叠加
#         axes[i, 3].imshow(after_img_np) # 先显示底图
#         # 再以半透明方式叠加显示热力图
#         im = axes[i, 3].imshow(change_magnitude_norm.cpu().numpy(), cmap='hot', alpha=0.5) 
#         axes[i, 3].set_title(f"Step {step_idx}: Heatmap Overlay")
#         axes[i, 3].axis('off')
#         fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

#     plt.tight_layout(pad=1.5, h_pad=3.0) # 调整布局防止标题重叠
    
#     # 保存图像
#     if not os.path.exists('transfer_learning'): os.mkdir('transfer_learning')
#     k = 0
#     while os.path.exists(f'transfer_learning/{category}_key_steps_details_v2_{k}.png'): k += 1
#     plt.savefig(f'transfer_learning/{category}_key_steps_details_v2_{k}.png')
#     plt.close()
#     print(f"Key steps details (v2) visualization saved to 'transfer_learning/{category}_key_steps_details_v2_{k}.png'")


# # 在 visualize_steps.py 中添加
# def plot_and_save_trajectory(trajectory: List[torch.Tensor], category: str, name: str):
#     """
#     将重建过程中的中间图像可视化并保存。
#     """
#     num_images = len(trajectory)
#     # 自动决定网格大小，让图片尽可能大
#     cols = int(np.ceil(np.sqrt(num_images)))
#     rows = int(np.ceil(num_images / cols))
    
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
#     axes = axes.flatten()
    
#     for i, img in enumerate(trajectory):
#         axes[i].imshow(show_tensor_image(img.cpu()))
#         axes[i].set_title(f"Step {i}", fontsize=8)
#         axes[i].axis('off')
        
#     # 隐藏多余的子图
#     for j in range(i + 1, len(axes)):
#         axes[j].axis('off')
        
#     plt.suptitle(f"Reconstruction Trajectory: {name}")
#     plt.tight_layout()
    
#     output_dir = 'swap/trajectories'
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{category}_{name}_trajectory.png")
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Saved trajectory visualization to {output_path}")

# 文件: visualize_steps.py
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import math
# from torchvision import transforms
# from typing import List

# def show_tensor_image(image: torch.Tensor):
#     """将单个 PyTorch Tensor 转换为可显示的 NumPy 图像数组。"""
#     if len(image.shape) == 4:
#         image = image[0, :, :, :]
    
#     # 自动处理 [0, 1] 和 [-1, 1] 两种范围的输入
#     if image.min() < 0:
#         image = (image + 1) / 2
        
#     reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         transforms.Lambda(lambda t: t.clamp(0, 1)),      # 确保值在 [0, 1] 范围内
#         transforms.Lambda(lambda t: t.cpu().numpy()),
#     ])
#     return reverse_transforms(image)

# def visualize_step_analysis_simple(step_norms: np.ndarray, category: str):
#     """绘制每个去噪步骤的特征范数（重要性）曲线图。"""
#     plt.figure(figsize=(15, 6))
#     plt.plot(range(len(step_norms)), step_norms, marker='.', linestyle='-')
#     plt.title(f'L2 Norm of Feature Vector at Each Denoising Step (Category: {category})', fontsize=16)
#     plt.xlabel('Denoising Step Index')
#     plt.ylabel('L2 Norm (Importance)')
#     plt.grid(True)
    
#     output_dir = 'transfer_learning'
#     os.makedirs(output_dir, exist_ok=True)
#     k = 0
#     while os.path.exists(os.path.join(output_dir, f'{category}_simple_step_analysis_{k}.png')):
#         k += 1
#     save_path = os.path.join(output_dir, f'{category}_simple_step_analysis_{k}.png')
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"Simple step analysis visualization saved to '{save_path}'")

# def visualize_key_step_details_v2(key_step_indices: List[int], images_before: List[torch.Tensor], images_after: List[torch.Tensor], diff_maps: List[torch.Tensor], category: str):
#     """为选定的关键步骤，并排显示去噪前、去噪后、变化幅度和热力图叠加。"""
#     num_key_steps = len(key_step_indices)
#     if num_key_steps == 0:
#         print("没有提供关键步骤，无法生成详细可视化图。")
#         return

#     fig, axes = plt.subplots(num_key_steps, 4, figsize=(20, 5 * num_key_steps))
#     if num_key_steps == 1:
#         axes = np.expand_dims(axes, axis=0)
        
#     for i, step_idx in enumerate(sorted(key_step_indices)):
#         before_img = images_before[step_idx]
#         after_img = images_after[step_idx]
#         diff_img = diff_maps[step_idx]

#         change_magnitude = torch.linalg.norm(diff_img.float(), ord=2, dim=0)
        
#         change_magnitude_norm = change_magnitude / change_magnitude.max() if change_magnitude.max() > 0 else change_magnitude

#         # 1. 去噪前
#         axes[i, 0].imshow(show_tensor_image(before_img))
#         axes[i, 0].set_title(f"Step {step_idx}: Before Denoising")
#         axes[i, 0].axis('off')

#         # 2. 去噪后
#         after_img_np = show_tensor_image(after_img)
#         axes[i, 1].imshow(after_img_np)
#         axes[i, 1].set_title(f"Step {step_idx}: After Denoising")
#         axes[i, 1].axis('off')

#         # 3. 变化幅度 (热力图)
#         axes[i, 2].imshow(change_magnitude_norm.cpu().numpy(), cmap='viridis')
#         axes[i, 2].set_title(f"Step {step_idx}: Change Magnitude")
#         axes[i, 2].axis('off')
        
#         # 4. 热力图叠加
#         axes[i, 3].imshow(after_img_np)
#         im = axes[i, 3].imshow(change_magnitude_norm.cpu().numpy(), cmap='hot', alpha=0.6) 
#         axes[i, 3].set_title(f"Step {step_idx}: Heatmap Overlay")
#         axes[i, 3].axis('off')
#         fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

#     plt.tight_layout(pad=1.5, h_pad=3.0)
    
#     output_dir = 'transfer_learning'
#     os.makedirs(output_dir, exist_ok=True)
#     k = 0
#     while os.path.exists(os.path.join(output_dir, f'{category}_key_steps_details_v2_{k}.png')):
#         k += 1
#     save_path = os.path.join(output_dir, f'{category}_key_steps_details_v2_{k}.png')
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Key steps details (v2) visualization saved to '{save_path}'")

# def plot_trajectory(trajectory: List[torch.Tensor], title: str, save_path: str):
#     """将一个完整的图像生成序列绘制成网格图并保存。"""
#     num_images = len(trajectory)
#     cols = int(math.ceil(math.sqrt(num_images)))
#     rows = int(math.ceil(num_images / cols))
    
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
#     axes = axes.flatten()
    
#     for i, img in enumerate(trajectory):
#         axes[i].imshow(show_tensor_image(img.cpu()))
#         axes[i].set_title(f"Step {i}", fontsize=10)
#         axes[i].axis('off')
        
#     for j in range(i + 1, len(axes)):
#         axes[j].axis('off')
        
#     fig.suptitle(title, fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.close()
#     print(f"去噪过程可视化图已保存至: {save_path}")

# 文件: visualize_steps.py
# 文件: visualize_steps.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from torchvision import transforms
from typing import List

def show_tensor_image(image: torch.Tensor):
    """将单个 PyTorch Tensor 转换为可显示的 NumPy 图像数组。"""
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    # 自动处理 [0, 1] 和 [-1, 1] 两种范围的输入
    if image.min() < 0:
        image = (image + 1) / 2
        
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t.clamp(0, 1)),      # 确保值在 [0, 1] 范围内
        transforms.Lambda(lambda t: t.cpu().numpy()),
    ])
    return reverse_transforms(image)

def visualize_step_analysis_simple(step_norms: np.ndarray, category: str):
    """绘制每个去噪步骤的特征范数（重要性）曲线图。"""
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(step_norms)), step_norms, marker='.', linestyle='-')
    plt.title(f'L2 Norm of Feature Vector at Each Denoising Step (Category: {category})', fontsize=16)
    plt.xlabel('Denoising Step Index')
    plt.ylabel('L2 Norm (Importance)')
    plt.grid(True)
    
    output_dir = 'transfer_learning'
    os.makedirs(output_dir, exist_ok=True)
    k = 0
    while os.path.exists(os.path.join(output_dir, f'{category}_simple_step_analysis_{k}.png')):
        k += 1
    save_path = os.path.join(output_dir, f'{category}_simple_step_analysis_{k}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Simple step analysis visualization saved to '{save_path}'")

def visualize_key_step_details_v2(key_step_indices: List[int], images_before: List[torch.Tensor], images_after: List[torch.Tensor], diff_maps: List[torch.Tensor], category: str):
    """为选定的关键步骤，并排显示去噪前、去噪后、变化幅度和热力图叠加。"""
    num_key_steps = len(key_step_indices)
    if num_key_steps == 0:
        print("没有提供关键步骤，无法生成详细可视化图。")
        return

    fig, axes = plt.subplots(num_key_steps, 4, figsize=(20, 5 * num_key_steps))
    if num_key_steps == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i, step_idx in enumerate(sorted(key_step_indices)):
        before_img = images_before[step_idx]
        after_img = images_after[step_idx]
        diff_img = diff_maps[step_idx]

        change_magnitude = torch.linalg.norm(diff_img.float(), ord=2, dim=0)
        
        change_magnitude_norm = change_magnitude / change_magnitude.max() if change_magnitude.max() > 0 else change_magnitude

        # 1. 去噪前
        axes[i, 0].imshow(show_tensor_image(before_img))
        axes[i, 0].set_title(f"Step {step_idx}: Before Denoising")
        axes[i, 0].axis('off')

        # 2. 去噪后
        after_img_np = show_tensor_image(after_img)
        axes[i, 1].imshow(after_img_np)
        axes[i, 1].set_title(f"Step {step_idx}: After Denoising")
        axes[i, 1].axis('off')

        # 3. 变化幅度 (热力图)
        axes[i, 2].imshow(change_magnitude_norm.cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title(f"Step {step_idx}: Change Magnitude")
        axes[i, 2].axis('off')
        
        # 4. 热力图叠加
        axes[i, 3].imshow(after_img_np)
        im = axes[i, 3].imshow(change_magnitude_norm.cpu().numpy(), cmap='hot', alpha=0.6) 
        axes[i, 3].set_title(f"Step {step_idx}: Heatmap Overlay")
        axes[i, 3].axis('off')
        fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=1.5, h_pad=3.0)
    
    output_dir = 'transfer_learning'
    os.makedirs(output_dir, exist_ok=True)
    k = 0
    while os.path.exists(os.path.join(output_dir, f'{category}_key_steps_details_v2_{k}.png')):
        k += 1
    save_path = os.path.join(output_dir, f'{category}_key_steps_details_v2_{k}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Key steps details (v2) visualization saved to '{save_path}'")

def plot_trajectory(trajectory: List[torch.Tensor], title: str, save_path: str):
    """将一个完整的图像生成序列绘制成网格图并保存。"""
    num_images = len(trajectory)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i, img in enumerate(trajectory):
        axes[i].imshow(show_tensor_image(img.cpu()))
        axes[i].set_title(f"Step {i}", fontsize=10)
        axes[i].axis('off')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"去噪过程可视化图已保存至: {save_path}")