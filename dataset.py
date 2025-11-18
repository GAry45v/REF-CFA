import os
import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10

# VisA的Dataset_maker (已修正为支持监督训练)
# class Dataset_maker(torch.utils.data.Dataset):
#     def __init__(self, root, category, config, is_train=True):
#         self.image_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),  
#                 transforms.ToTensor(), # Scales data into [0,1] 
#                 transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
#             ]
#         )
#         self.config = config
#         self.mask_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),
#                 transforms.ToTensor(), # Scales data into [0,1] 
#             ]
#         )
        
#         self.root = root
#         self.category = category
#         self.is_train = is_train

#         # --- VisA 路径逻辑修改 ---
#         image_base_path = os.path.join(root, category, "Data", "Images")
        
#         # 定义一个辅助函数来查找所有常见图像文件
#         def find_image_files(path):
#             return sorted(glob(os.path.join(path, "*.png"))) + \
#                    sorted(glob(os.path.join(path, "*.jpg"))) + \
#                    sorted(glob(os.path.join(path, "*.JPG")))

#         if is_train:
#             # (如您所愿) 训练集现在包含 "Normal" + "Anomaly"
#             normal_files = find_image_files(os.path.join(image_base_path, "Normal"))
#             anomaly_files = find_image_files(os.path.join(image_base_path, "Anomaly"))
            
#             self.image_files = normal_files + anomaly_files
            
#         else:
#             # 测试集 = "Normal" + "Anomaly" 文件夹中的图像
#             normal_files = find_image_files(os.path.join(image_base_path, "Normal"))
#             anomaly_files = find_image_files(os.path.join(image_base_path, "Anomaly"))
            
#             self.image_files = normal_files + anomaly_files
#         # --- 结束修改 ---

#     def __getitem__(self, index):
#         image_file = self.image_files[index]
        
#         # 使用 .convert('RGB') 来确保图像总是3通道
#         image = Image.open(image_file).convert('RGB')
#         image = self.image_transform(image)
        
#         if(image.shape[0] == 1):
#             image = image.expand(3, self.config.data.image_size, self.config.data.image_size)

#         if self.is_train:
#             # --- (核心修改) ---
#             # 检查图像的父文件夹名来确定标签
#             is_good = Path(image_file).parent.name == "Normal"
            
#             if is_good:
#                 label = 'good'
#             else:
#                 label = 'defective'
                
#             return image, label # 返回 (图像, 标签)
#             # --------------------
            
#         else:
#             # (测试逻辑保持不变, 它已经正确处理 Normal/Anomaly)
#             is_good = Path(image_file).parent.name == "Normal"
            
#             if self.config.data.mask:
#                 if is_good:
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'good'
#                 else : 
#                     label = 'defective'
#                     filename = Path(image_file).stem
#                     mask_path = os.path.join(self.root, self.category, "Data", "Masks", f"{filename}.png")
                    
#                     if os.path.exists(mask_path):
#                         target = Image.open(mask_path)
#                         target = self.mask_transform(target)
#                     else:
#                         print(f"警告: 掩码文件未找到: {mask_path}")
#                         target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#             else:
#                 if is_good:
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'good'
#                 else :
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'defective'
                
#             return image, target, label

#     def __len__(self):
#         return len(self.image_files)

# MVtec的Dataset_Maker
class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        if is_train:
            if category:
                # self.image_files = glob(
                #     os.path.join(root, category, "train", "good", "*.png")
                # )
                 self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(
                    os.path.join(root, "train", "good", "*.png")
                )
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    if self.config.data.name == 'MVTec':
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/").replace(
                                ".png", "_mask.png"
                            )
                        )
                    else:
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/"))
                    target = self.mask_transform(target)
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'defective'
                
            return image, target, label

    def __len__(self):
        return len(self.image_files)


#DAGM
# class Dataset_maker(torch.utils.data.Dataset):
#     """
#     用于加载 DAGM 数据集的 Dataset 类。
    
#     文件结构 (基于您的截图):
#     - .../DAGM_dataset/
#         - Class1/       (正常图片, e.g., 0001.png)
#         - Class1_def/   (异常图片, e.g., 0001.png 和 0001_mask.png)
#         - ...

#     训练/测试逻辑 (基于您的要求):
#     - 训练 (is_train=True): 
#         只加载 'category' (e.g., Class1) 中的图片。
#         返回: (image, 'good')
#     - 测试 (is_train=False): 
#         加载 'category' (e.g., Class1) 和 'category_def' (e.g., Class1_def) 中的图片。
#         返回: (image, target_mask, label_str)
#     """
#     def __init__(self, root, category, config, is_train=True):
#         """
#         Args:
#             root (str): 数据集根目录 (e.g., .../DAGM_dataset)
#             category (str): 要加载的类别 (e.g., "Class1")
#             config (OmegaConf): 包含 config.data.image_size 的配置对象
#             is_train (bool): 切换训练和测试模式
#         """
#         self.root = root
#         self.category = category
#         self.is_train = is_train
#         self.config = config

#         # 1. 定义与模板一致的图像变换
#         self.image_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),
#                 transforms.ToTensor(), # 缩放到 [0,1]
#                 transforms.Lambda(lambda t: (t * 2) - 1) # 缩放到 [-1, 1]
#             ]
#         )
#         # 2. 定义与模板一致的掩码变换
#         self.mask_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),
#                 transforms.ToTensor(), # 缩放到 [0,1]
#             ]
#         )

#         # 3. (核心) 根据 is_train 标志构建文件列表
#         self.image_files = [] # 列表将存储元组 (image_path, label_str, mask_path_or_None)
        
#         # 确定正常和异常文件夹的路径
#         normal_dir = os.path.join(self.root, self.category)
        
#         valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

#         if self.is_train:
#             # --- 训练逻辑：只加载 'category' (正常) ---
#             normal_paths = glob.glob(os.path.join(normal_dir, "*.*"))
#             for img_path in normal_paths:
#                 if img_path.lower().endswith(valid_extensions):
#                     # (路径, 标签, 掩码路径)
#                     self.image_files.append((img_path, 'good', None))
#         else:
#             # --- 测试逻辑：加载 'category' (正常) 和 'category_def' (异常) ---
            
#             # A. 加载正常文件
#             normal_paths = glob.glob(os.path.join(normal_dir, "*.*"))
#             for img_path in normal_paths:
#                 if img_path.lower().endswith(valid_extensions):
#                     self.image_files.append((img_path, 'good', None))
            
#             # B. 加载异常文件
#             defective_dir = os.path.join(self.root, self.category + "_def")
#             all_defective_files = glob.glob(os.path.join(defective_dir, "*.*"))
            
#             # 区分图像和掩码
#             image_paths_def = []
#             mask_map = {} # (用于快速查找掩码)
            
#             for f_path in all_defective_files:
#                 if not f_path.lower().endswith(valid_extensions):
#                     continue # 跳过非图像文件
                
#                 f_name_lower = os.path.basename(f_path).lower()
                
#                 if '_mask' in f_name_lower:
#                     mask_map[f_name_lower] = f_path
#                 else:
#                     image_paths_def.append(f_path)
            
#             # C. 匹配图像和掩码
#             for img_path in image_paths_def:
#                 base_name = os.path.basename(img_path)
#                 name_part, ext = os.path.splitext(base_name)
                
#                 # 假设掩码命名为: "0001.png" -> "0001_mask.png"
#                 mask_name_lower = (name_part + "_mask" + ext).lower()
#                 mask_path = mask_map.get(mask_name_lower, None)
                
#                 if mask_path is None:
#                     print(f"警告: 找不到异常图像 {img_path} 对应的掩码文件 {mask_name_lower}")
                
#                 self.image_files.append((img_path, 'defective', mask_path))

#         print(f"--- DagmDataset ({self.category}, is_train={self.is_train}) ---")
#         print(f"Total images loaded: {len(self.image_files)}")
        
#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):
#         # 1. 从文件列表中获取信息
#         image_file, label, mask_file = self.image_files[index]
        
#         # 2. 加载和转换图像
#         image = Image.open(image_file).convert("RGB") # 转换为RGB以确保3通道
#         image = self.image_transform(image)
        
#         # (来自模板的代码：以防万一加载了灰度图)
#         if(image.shape[0] == 1):
#             image = image.expand(3, self.config.data.image_size, self.config.data.image_size)

#         if self.is_train:
#             # --- 训练逻辑：只返回 (image, label_str) ---
#             return image, label
#         else:
#             # --- 测试逻辑：返回 (image, target_mask, label_str) ---
            
#             if label == 'good':
#                 # 正常样本，创建空掩码
#                 target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#             else:
#                 # 异常样本，加载掩码
#                 if mask_file is not None and os.path.exists(mask_file):
#                     target = Image.open(mask_file).convert("L") # 以灰度模式加载掩码
#                     target = self.mask_transform(target)
#                 else:
#                     # 即使是异常，如果没找到掩码也返回空掩码
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            
#             return image, target, label

# class Dataset_maker(torch.utils.data.Dataset):
#     def __init__(self, root, category, config, is_train=True):
#         self.image_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),
#                 transforms.ToTensor(), # Scales data into [0,1]
#                 transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
#             ]
#         )
#         self.config = config
#         self.mask_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),
#                 transforms.ToTensor(), # Scales data into [0,1]
#             ]
#         )

#         # --- MODIFICATION START ---
#         # Define image extensions to look for
#         image_extensions = ['png', 'jpg', 'jpeg']
#         self.image_files = []
        
#         if is_train:
#             # Path for training data (includes all subdirectories like 'good', 'defective', etc.)
#             path_pattern = os.path.join(root, category, "train", "*") if category else os.path.join(root, "train", "*")
#             # Find all image files with the specified extensions
#             for ext in image_extensions:
#                 self.image_files.extend(glob(os.path.join(path_pattern, f"*.{ext.lower()}")))
#                 self.image_files.extend(glob(os.path.join(path_pattern, f"*.{ext.upper()}")))
#         else:
#             # Path for test data
#             path_pattern = os.path.join(root, category, "test", "*") if category else os.path.join(root, "test", "*")

#             # Find all image files with the specified extensions
#             for ext in image_extensions:
#                 self.image_files.extend(glob(os.path.join(path_pattern, f"*.{ext.lower()}")))
#                 self.image_files.extend(glob(os.path.join(path_pattern, f"*.{ext.upper()}")))
#         # --- MODIFICATION END ---
        
#         self.is_train = is_train

#     def __getitem__(self, index):
#         image_file = self.image_files[index]
#         # Convert image to RGB to handle grayscale images
#         image = Image.open(image_file).convert("RGB")
#         image = self.image_transform(image)

#         if self.is_train:
#             if os.path.dirname(image_file).endswith("good"):
#                 label = 'good'
#             else:
#                 label = 'defective'
#             return image, label
#         else:
#             if self.config.data.mask:
#                 if os.path.dirname(image_file).endswith("good"):
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'good'
#                 else:
#                     # Logic for finding mask files needs to handle different extensions too
#                     base_name = Path(image_file).stem
#                     mask_path_png = image_file.replace("/test/", "/ground_truth/").replace(f"{base_name}{Path(image_file).suffix}", f"{base_name}_mask.png")
                    
#                     if self.config.data.name == 'MVTec' and os.path.exists(mask_path_png):
#                          target = Image.open(mask_path_png)
#                     else:
#                         # Fallback for other datasets or if _mask.png doesn't exist
#                         ground_truth_dir = os.path.dirname(image_file).replace("/test/", "/ground_truth/")
#                         # Assuming ground truth has the same name and extension
#                         mask_path_original_ext = os.path.join(ground_truth_dir, os.path.basename(image_file))
#                         target = Image.open(mask_path_original_ext)

#                     target = self.mask_transform(target)
#                     label = 'defective'
#             else:
#                 if os.path.dirname(image_file).endswith("good"):
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'good'
#                 else:
#                     target = torch.ones([1, image.shape[-2], image.shape[-1]]) # Use ones for defective GT if no mask
#                     label = 'defective'
                
#             return image, target, label

#     def __len__(self):
#         return len(self.image_files)

# 文件: dataset.py (推荐的修改版)
# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from pathlib import Path
# import numpy as np
# import glob # 确保导入 glob

# class Dataset_maker(Dataset):
#     def __init__(self, root, category, config, is_train=False, load_masks=False): # --- 1. 添加 load_masks ---
#         self.config = config
#         self.root = root
#         self.category = category
#         self.is_train = is_train
#         self.load_masks = load_masks # --- 2. 保存状态 ---

#         self.image_files = self._get_image_files()
        
#         # 图像变换
#         self.image_transform = transforms.Compose([
#             transforms.Resize((config.data.image_size, config.data.image_size)),
#             transforms.ToTensor(), # 缩放到 [0, 1]
#             transforms.Lambda(lambda t: (t * 2) - 1) # 缩放到 [-1, 1]
#         ])
        
#         # 掩码变换
#         self.mask_transform = transforms.Compose([
#             transforms.Resize((config.data.image_size, config.data.image_size), 
#                               interpolation=transforms.InterpolationMode.NEAREST), # --- 3. 使用 NEAREST ---
#             transforms.ToTensor(), # 缩放到 [0, 1]
#         ])

#     def _get_image_files(self):
#         # --- 4. 修正路径逻辑 ---
#         if self.is_train:
#             # 训练时，只加载 train/good 目录
#             search_path = os.path.join(self.root, self.category, "train", "good", "*.png")
#         else:
#             # 测试时，加载 test/ 目录下的所有子文件夹
#             search_path = os.path.join(self.root, self.category, "test", "*", "*.png")
            
#         image_files = glob.glob(search_path)
#         return sorted(image_files)

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):
#         image_path = self.image_files[index]
        
#         # 1. 加载图像
#         image = Image.open(image_path).convert('RGB')
#         image = self.image_transform(image)
#         # (您的单通道转三通道逻辑是好的，如果需要可以保留)
#         if(image.shape[0] == 1):
#             image = image.expand(3, self.config.data.image_size, self.config.data.image_size)

#         # 2. 获取标签
#         # (使用 Pathlib 更健壮)
#         parent_dir_name = Path(image_path).parent.name
#         if parent_dir_name == "good":
#             label = "good"
#         else:
#             label = "defective" # 统一所有异常为 'defective'
        
#         # --- 5. 核心逻辑：根据 load_masks 切换 ---
        
#         if self.load_masks:
#             # --- 模式A: 用于有监督定位训练 ---
#             # 返回 (image, mask, label)
            
#             mask = None
#             if label == "good":
#                 # “好”样本，返回全零掩码
#                 mask = torch.zeros(1, self.config.data.image_size, self.config.data.image_size)
#             else:
#                 # “坏”样本，加载真实掩码
#                 # 假设 MVTec 结构: .../test/crack/001.png -> .../ground_truth/crack/001_mask.png
#                 mask_path = image_path.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
                
#                 if not os.path.exists(mask_path):
#                     # Fallback (以防万一，例如 VisA 或您的版本没有 _mask 后缀)
#                      mask_path = image_path.replace("/test/", "/ground_truth/")
                     
#                 if os.path.exists(mask_path):
#                     mask_image = Image.open(mask_path).convert('L')
#                     mask = self.mask_transform(mask_image)
#                     mask = (mask > 0.5).float() # 二值化
#                 else:
#                     print(f"警告: 找不到掩码 {mask_path}")
#                     mask = torch.zeros(1, self.config.data.image_size, self.config.data.image_size)
            
#             return image, mask, label

#         else:
#             # --- 模式B: 用于 UNet 训练 或 分类器推理 ---
#             # 保持 (data, _, label) 的格式
#             # 我们返回 image_path 作为第二个元素，您的旧代码返回 target，这里统一
            
#             # (如果您的代码严格依赖 (image, label) 或 (image, target, label)，
#             # 我们可以保留您原来的 if self.is_train 判断)
            
#             # --- 让我们严格匹配您的旧逻辑：---
#             if self.is_train:
#                 # 用于 train_unet: 返回 (image, label)
#                 return image, label
#             else:
#                 # 用于 detection: 返回 (image, target_placeholder, label)
#                 # detection 需要 (image, target, label)
#                 if label == 'good':
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                 else:
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]]) # 分类器不需要真实掩码
#                 return image, target, label