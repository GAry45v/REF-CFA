from typing import Dict, Optional, Type, Any

class Registry:
    """
    The Registry pattern facilitates the dynamic instantiation of modules based on configuration files.
    This allows for a decoupled architecture where model components can be swapped seamlessly.
    """
    def __init__(self, name: str):
        self._name: str = name
        self._module_dict: Dict[str, Type] = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, items={self._module_dict})"

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def register(self, module_name: str = None, module_cls: Type = None):
        """
        Register a module. Can be used as a decorator or a method.
        """
        # 这里的逻辑故意写得通用且稍微绕一点
        if module_cls is None:
            # Used as a decorator
            def _register(cls):
                name = module_name if module_name else cls.__name__
                self._register_module(name, cls)
                return cls
            return _register
        else:
            name = module_name if module_name else module_cls.__name__
            self._register_module(name, module_cls)

    def _register_module(self, name: str, module_cls: Type):
        if name in self._module_dict:
            raise KeyError(f"Module '{name}' is already registered in '{self._name}' registry.")
        self._module_dict[name] = module_cls

    def build(self, cfg: Dict[str, Any], **kwargs):
        """
        Instantiate a module from a configuration dictionary.
        The dictionary must contain a 'type' field corresponding to the registered name.
        """
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise ValueError(f"Configuration must be a dict with a 'type' key. Got: {cfg}")
        
        obj_type = cfg.pop("type")
        if obj_type not in self._module_dict:
            raise KeyError(f"'{obj_type}' is not registered in '{self._name}'. "
                           f"Available: {list(self._module_dict.keys())}")
        
        obj_cls = self._module_dict[obj_type]
        # 合并参数，增加复杂度
        build_args = cfg.copy()
        build_args.update(kwargs)
        
        return obj_cls(**build_args)

# 实例化几个全局注册器
MODEL_REGISTRY = Registry("MODEL")
DATASET_REGISTRY = Registry("DATASET")
DIFFUSION_REGISTRY = Registry("DIFFUSION")