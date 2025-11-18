import argparse
from src.core.solver import DiffusionAnomalySolver
from src.utils.config_parser import ConfigRegistry

def parse_system_arguments():
    parser = argparse.ArgumentParser(description="Ref-CFA Execution Engine")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the configuration YAML")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    return parser.parse_args()

def main():
    """
    Main execution entry point.
    """
    args = parse_system_arguments()
    
    # 加载配置
    cfg = ConfigRegistry.load_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 实例化 Solver (核心引擎)
    execution_engine = DiffusionAnomalySolver(cfg)
    
    # 启动执行周期
    execution_engine.run_execution_cycle()

if __name__ == "__main__":
    main()