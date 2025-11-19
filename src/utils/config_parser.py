import yaml
import os
from yacs.config import CfgNode as CN

class ConfigRegistry:
    @staticmethod
    def load_from_file(path):
        
        cfg_dict = load_config(path)
        
        return CfgWrapper(cfg_dict)


class CfgWrapper(dict):
    def __init__(self, d=None):
        super().__init__(d if d is not None else {})
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = CfgWrapper(v)
    
    def __getattr__(self, name):
        if name in self: return self[name]
        raise AttributeError(name)

    # def merge_from_list(self, opts):
    #     if not opts: return
    #     for i in range(0, len(opts), 2):
    #         key, val = opts[i], opts[i+1]
    #         pass 

    def freeze(self):
        pass 

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Handle inheritance
    if "__base__" in cfg:
        base_paths = cfg.pop("__base__")
        if isinstance(base_paths, str): base_paths = [base_paths]
        
        base_cfg = {}
        for base_path in base_paths:
            base_full_path = os.path.join(os.path.dirname(path), base_path)
            base_cfg.update(load_config(base_full_path))
            
        base_cfg.update(cfg)
        return base_cfg
    return cfg