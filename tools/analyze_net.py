import argparse
import os
import sys
import yaml
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import ConfigRegistry
from src.core.analyzer import AnalysisEngine
from src.utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description="Ref-CFA Analysis & Tooling")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Optional explicit checkpoint path")
    
    # Modes
    parser.add_argument("--analyze_transformer", action="store_true", help="Run Transformer-based analysis")
    parser.add_argument("--force_train_transformer", action="store_true", help="Retrain the transformer")
    
    parser.add_argument("--train_localization", action="store_true", help="Train supervised localization on source")
    parser.add_argument("--run_supervised_localization", action="store_true", help="Run supervised inference on target")
    
    parser.add_argument("--analyze_residuals", action="store_true", help="Visualize step-wise residuals")
    
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Config overrides")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Configuration
    cfg = ConfigRegistry.load_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
        
    # Override checkpoint if provided via CLI
    if args.checkpoint:
        cfg.solver.resume_checkpoint = args.checkpoint
        
    # 2. Initialize Engine
    logger = Logger.get_logger("RefCFA.Analysis")
    logger.info(f"Initializing Analysis Engine for experiment: {cfg.system.experiment_name}")
    
    analyzer = AnalysisEngine(cfg)
    
    # 3. Execute Tasks
    if args.analyze_transformer:
        analyzer.run_transformer_analysis(force_train=args.force_train_transformer)
        
    if args.train_localization:
        analyzer.train_supervised_localization()
        
    if args.run_supervised_localization:
        analyzer.run_supervised_inference()
        
    if args.analyze_residuals:
        analyzer.analyze_residuals()
        
    if not any([args.analyze_transformer, args.train_localization, 
                args.run_supervised_localization, args.analyze_residuals]):
        logger.warning("No analysis mode selected. Use flags like --analyze_transformer")

if __name__ == "__main__":
    main()