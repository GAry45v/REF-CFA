# 简单的配置合并逻辑示例 (你可以加在你的工具脚本中)
import yaml
import os

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Handle inheritance
    if "__base__" in cfg:
        base_paths = cfg.pop("__base__")
        if isinstance(base_paths, str): base_paths = [base_paths]
        
        base_cfg = {}
        for base_path in base_paths:
            # Resolve relative path
            base_full_path = os.path.join(os.path.dirname(path), base_path)
            base_cfg.update(load_config(base_full_path))
            
        # Merge (simple recursive update recommended for real usage)
        base_cfg.update(cfg)
        return base_cfg
    return cfg