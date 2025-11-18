import torch
import numpy as np
import os
import argparse
from unet import UNetModel
from omegaconf import OmegaConf
from train import trainer
from transformer import DDAD_Transformer_Analysis, SupervisedLocalizationModule
# from dataset import Dataset_maker
from dataset import Dataset_maker
from torch.utils.data import DataLoader
from reconstruction import Reconstruction
from pathlib import Path
from torch.utils.data import Subset  # 您应该已经添加了这行


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

def build_model(config):
    if config.model.DDADS:
        unet = UNetModel(config.data.image_size, 32, dropout=0.3, n_heads=2 ,in_channels=config.data.input_channel)
    else:
        unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.input_channel)
    return unet

# ... (在 build_model 函数之后) ...

def load_pretrained_unet_and_recon(config):
    """
    辅助函数：加载预训练的 UNet checkpont 
    并初始化 Reconstruction 模块。
    """
    unet = build_model(config)
    clean_object_name = config.data.category.split("_")[0]
    checkpoint_path = os.path.join(os.getcwd(), config.model.checkpoint_dir, clean_object_name, str(config.model.load_chp))
    # checkpoint_path = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp))
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)
    unet.to(config.model.device)
    unet.eval()
    
    # 创建 Reconstruction 模块
    recon_module = Reconstruction(unet, config)
    
    return unet, recon_module

def train_unet(config):
    torch.manual_seed(42)
    np.random.seed(42)
    unet = build_model(config)
    print("UNet Num params: ", sum(p.numel() for p in unet.parameters()))
    unet = unet.to(config.model.device)
    unet.train()
    unet = torch.nn.DataParallel(unet)
    trainer(unet, config.data.category, config)

# def detection_and_analysis(config, force_train_transformer):
#     unet = build_model(config)
#     # clean_object_name = config.data.category.split("_")[0]
#     # checkpoint_path = os.path.join(os.getcwd(), config.model.checkpoint_dir, clean_object_name, str(config.model.load_chp))
#     checkpoint_path = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp))
#     print(f"Loading checkpoint from: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path)
    
#     unet = torch.nn.DataParallel(unet)
#     unet.load_state_dict(checkpoint)
#     unet.to(config.model.device)
#     unet.eval()
    
#     # 使用新的分析类
#     ddad_analyzer = DDAD_Transformer_Analysis(unet, config)
#     ddad_analyzer(force_train=force_train_transformer) # 将参数传递下去

# def run_localization(config, args):
#     """
#     (新函数)
#     专门用于加载模型并运行定位。
#     """
#     unet = build_model(config)
#     clean_object_name = config.data.category.split("_")[0]
#     checkpoint_path = os.path.join(os.getcwd(), config.model.checkpoint_dir, clean_object_name, str(config.model.load_chp))
#     print(f"Loading checkpoint from: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path)
    
#     unet = torch.nn.DataParallel(unet)
#     unet.load_state_dict(checkpoint)
#     unet.to(config.model.device)
#     unet.eval()
    
#     ddad_analyzer = DDAD_Transformer_Analysis(unet, config)
    
#     # 调用新的定位函数
#     # 调用新的定位函数，并传递新参数
#     ddad_analyzer.run_localization(
#         num_samples=args.num_loc_samples,
#         threshold=args.loc_threshold,
#         use_final_residual=args.use_final_residual  # <--- (新) 传递参数
#     )

def detection_and_analysis(config, force_train_transformer):
    # --- 用辅助函数替换模型加载代码 ---
    unet, _ = load_pretrained_unet_and_recon(config)
    # ------------------------------------
    
    # (其余代码不变)
    # 使用新的分析类
    ddad_analyzer = DDAD_Transformer_Analysis(unet, config)
    ddad_analyzer(force_train=force_train_transformer) # 将参数传递下去

def run_localization(config, args):
    """
    (函数不变)
    专门用于加载模型并运行 (无监督) 定位。
    """
    # --- 用辅助函数替换模型加载代码 ---
    unet, _ = load_pretrained_unet_and_recon(config)
    # ------------------------------------
    
    ddad_analyzer = DDAD_Transformer_Analysis(unet, config)
    
    # (其余代码不变)
    ddad_analyzer.run_localization(
        num_samples=args.num_loc_samples,
        threshold=args.loc_threshold,
        use_final_residual=args.use_final_residual 
    )

def train_supervised_localization(config):
    """
    (新函数 - 已修改)
    在源域上训练有监督定位分割器。
    (此版本经过修改，只在“异常”样本上训练，以解决模型“躺平”问题)
    """
    print('--- 开始训练有监督定位模型 (源域 - 仅异常样本) ---')
    # 加载 UNet 和 Reconstruction 模块
    # 我们不训练 UNet，只用它来提取特征
    _, recon_module = load_pretrained_unet_and_recon(config)
    
    # --- 关键步骤：加载源域数据集 (带掩码) ---
    print(f"加载源域数据集 (带掩码): {config.data.source_category}")
    try:
        # 1. 首先，加载完整的测试数据集
        source_dataset = Dataset_maker(
            root=config.data.source_data_dir,
            category=config.data.source_category,
            config=config, 
            is_train=False, 
            load_masks=True # <-- 必须为 True
        )
        
        # --- (关键修改：开始) ---
        # 筛选出 "defective" (非 "good") 样本的索引
        print("正在筛选[异常样本]以进行专项训练...")
        defective_indices = []
        
        # 检查 dataset 是否有 image_files 属性 (我们的版本有)
        if not hasattr(source_dataset, 'image_files'):
             print("!!! 错误：Dataset_maker 没有 'image_files' 属性。无法筛选样本。!!!")
             print("请确保您使用的是我们之前讨论过的、包含 'image_files' 属性的 dataset.py 版本。")
             return

        for i, file_path in enumerate(source_dataset.image_files):
            # 检查父文件夹名是否 *不是* "good"
            if Path(file_path).parent.name.lower() != "good":
                defective_indices.append(i)
        
        print(f"在 {len(source_dataset)} 个总样本中，找到了 {len(defective_indices)} 个异常样本。")

        if not defective_indices:
            print(f"!!! 错误：在 {config.data.source_category} 中找不到任何异常样本进行训练！!!!")
            return
            
        # 3. 使用 Subset 只保留异常样本
        train_subset = Subset(source_dataset, defective_indices)
        # --- (关键修改：结束) ---

        source_loader = DataLoader(
            train_subset, # <-- (注意) 现在使用 train_subset
            batch_size=config.data.train_batch_size, 
            shuffle=True, 
            num_workers=config.model.num_workers
        )
        print(f"源域数据集加载成功，将使用 {len(train_subset)} 个异常样本进行训练。")
    
    except Exception as e:
        print(f"!!! 错误：无法加载源域数据集。!!!")
        print(f"请确保您的 config.yaml 中定义了 'data.source_data_dir' 和 'data.source_category'。")
        print(f"并且您的 Dataset_maker 实现了 `load_masks=True` 功能。")
        print(f"原始错误: {e}")
        return

    # 初始化定位模块
    localizer = SupervisedLocalizationModule(recon_module, config, config.model.device)
    
    # 从 config 中获取训练周期数，如果不存在则默认为 20
    num_epochs = getattr(config.model, 'loc_epochs', 20)
    
    # 开始训练
    localizer.train(source_loader, num_epochs=num_epochs)

def run_supervised_localization(config, args):
    """
    (新函数)
    在目标域上运行有监督定位 (推理)。
    """
    print('--- 开始在目标域上运行有监督定位 (推理) ---')
    # 加载 UNet 和 Reconstruction 模块
    _, recon_module = load_pretrained_unet_and_recon(config)
    
    # --- 加载目标域数据集 (即您的标准测试集) ---
    # (这不需要掩码)
    print(f"加载目标域数据集 (推理): {config.data.category}")
    category = config.data.category+"_new"
    target_dataset = Dataset_maker(
        root=config.data.data_dir,
        category=category,
        config=config, 
        is_train=False,
        load_masks=False # 推理时不需要掩码
    )
    # 定位时 batch_size 通常为 1
    target_loader = DataLoader(
        target_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config.model.num_workers
    )

    # 初始化定位模块
    localizer = SupervisedLocalizationModule(recon_module, config, config.model.device)
    
    # 运行推理
    localizer.localize_on_target_domain(
        target_loader, 
        num_samples=args.num_sup_loc_samples, # 使用新的参数
        threshold=args.sup_loc_threshold  # 使用新的参数
    )

def run_residual_analysis(config):
    """
    (新) 启动逐步残差分析。
    此函数正确地将数据加载和分析任务委托给 DDAD_Transformer_Analysis 类，
    以避免 'module object is not callable' 错误。
    """
    print('Running Step-wise Residual Analysis...') # <-- (这是 main 中的日志)
    
    # 1. 使用你的辅助函数加载 UNet 和 Recon 模块
    try:
        unet, recon_module = load_pretrained_unet_and_recon(config)
    except Exception as e:
        print(f"--- ❌ 错误: 加载预训练模型失败 ---")
        print(f"请确保 config.model.load_chp 指向一个有效的 .pth 文件。")
        print(f"原始错误: {e}")
        return

    # 2. 实例化分析器
    # (这一步会触发 DDAD_Transformer_Analysis 的 __init__，
    #  它将在 ddad_transformer.py 内部安全地加载 Dataset_maker)
    try:
        print("Initializing DDAD_Transformer_Analysis (which loads the dataset)...")
        # (重要: 确保 DDAD_Transformer_Analysis 的 __init__ 
        #  可以正确处理 config.data.mask=True)
        analyzer = DDAD_Transformer_Analysis(unet, config)
        print("--- ✅ Analyzer and Dataset initialized successfully.")
    
    except Exception as e:
        print(f"--- ❌ 错误: 在 DDAD_Transformer_Analysis 初始化时失败 ---")
        print(f"  (这仍然可能是 Dataset_maker 的问题，但现在它发生在正确的文件中)")
        print(f"  原始错误: {e}")
        return

    # 3. 调用 *类方法* 来执行分析
    # (我们将在下一步中把这个方法添加到你的类中)
    if hasattr(analyzer, 'analyze_step_residuals'):
        # 传递 recon_module，因为这个方法需要它
        analyzer.analyze_step_residuals() 
    else:
        print("--- ❌ 错误: DDAD_Transformer_Analysis 类中缺少 'analyze_step_residuals' 方法。 ---")
        print("--- 请将该方法添加到 ddad_transformer.py 中。 ---")

def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD_Transformer_Analysis')
    cmdline_parser.add_argument('-cfg', '--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), help='config file')
    
    # 第 1 组: UNet 训练
    cmdline_parser.add_argument('--train_unet', action='store_true', help='训练 U-Net 模型')
    
    # 第 2 组: 异常检测 (Transformer 分类)
    cmdline_parser.add_argument('--detection', action='store_true', help='运行异常检测和 Transformer 分析')
    cmdline_parser.add_argument('--force_train_transformer', action='store_true', help='强制重新训练 Transformer 模型。')
    
    # 第 3 组: 无监督定位 (Heatmap)
    cmdline_parser.add_argument('--localize', action='store_true', help='(无监督) 运行异常定位 (使用 diff_maps)')
    cmdline_parser.add_argument('--num_loc_samples', type=int, default=5, help='无监督定位的样本数')
    cmdline_parser.add_argument('--loc_threshold', type=float, default=0.3, help='无监督定位的阈值')
    cmdline_parser.add_argument('--use_final_residual', action='store_true', 
                                help='(无监督) 使用最终残差图代替聚合 diff_maps')
    
    # --- 开始：添加新参数 ---
    # 第 4 组: 有监督定位 (新模块)
    cmdline_parser.add_argument('--train_localization', action='store_true', 
                                help='(有监督) 在源域上训练定位分割器 (需要掩码)')
    cmdline_parser.add_argument('--run_supervised_localization', action='store_true', 
                                help='(有监督) 在目标域上运行训练好的定位分割器')
    cmdline_parser.add_argument('--num_sup_loc_samples', type=int, default=10, 
                                help='(有监督) 定位时要可视化的目标样本数')
    cmdline_parser.add_argument('--sup_loc_threshold', type=float, default=0.5, 
                                help='(有监督) 预测掩码的二值化阈值')
    # --- (新) 第 5 组: 逐步残差分析 ---
    cmdline_parser.add_argument('--analyze_residuals', action='store_true', 
                                help='(新) 运行特定样本的逐步残差分析 (需要 data.mask=True)')
    # --- 结束：添加新参数 ---

    return cmdline_parser

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # --- (新) 配置加载逻辑 ---
    # 1. 获取解析器
    parser = parse_args()
    args, unknown_args = parser.parse_known_args()
    
    # 3. 从文件加载基础配置
    config = OmegaConf.load(args.config)
    
    # 4. 从命令行加载 "未知" 参数作为覆盖
    cli_config = OmegaConf.from_cli(unknown_args)
    
    # 5. 合并配置 (命令行的优先级更高)
    config = OmegaConf.merge(config, cli_config)
    # --- (结束新逻辑) ---
    
    print("--- Configuration ---")
    # (现在 config 包含了合并后的值)
    print(f"Class: {config.data.category}, w: {config.model.w}, v: {config.model.v}, load_chp: {config.model.load_chp}")
    print("---------------------")

    # 设置随机种子
    torch.manual_seed(config.model.seed)
    np.random.seed(config.model.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.model.seed)

    # --- (您已有的逻辑) ---
    if args.train_unet:
        print('Training U-Net...')
        train_unet(config)
        
    if args.detection:
        print('Running Detection and Transformer Analysis...')
        detection_and_analysis(config, args.force_train_transformer)

    if args.localize:
        print('Running Anomaly Localization (Heatmap)...')
        run_localization(config, args) # 传递所有 args

    # --- (开始：您缺少的逻辑) ---
    # 添加对新功能的支持
    if args.train_localization:
        print('Starting SUPERVISED Localization Training...')
        train_supervised_localization(config)
        
    if args.run_supervised_localization:
        print('Running SUPERVISED Localization on Target Domain...')
        run_supervised_localization(config, args)

    if args.analyze_residuals:
        print('Running Step-wise Residual Analysis...')
        run_residual_analysis(config) # 只需要 config
    # --- (结束：您缺少的逻辑) ---