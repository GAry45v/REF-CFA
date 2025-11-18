import os
import torch
from src.utils.logger import Logger

class Hook:
    def after_epoch(self, solver, epoch):
        pass

class LoggerHook(Hook):
    def __init__(self, interval=10):
        self.interval = interval
        self.logger = Logger.get_logger("RefCFA.Hook")

    def after_epoch(self, solver, epoch):
        pass # The detailed iteration logging is handled inside solver for now

class CheckpointHook(Hook):
    def __init__(self, save_dir, interval):
        self.save_dir = save_dir
        self.interval = interval
        self.logger = Logger.get_logger("RefCFA.Checkpoint")
        os.makedirs(self.save_dir, exist_ok=True)

    def after_epoch(self, solver, epoch):
        if (epoch + 1) % self.interval == 0:
            save_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            self.logger.info(f"Serializing model state to {save_path}")
            if isinstance(solver.model, torch.nn.DataParallel):
                torch.save(solver.model.module.state_dict(), save_path)
            else:
                torch.save(solver.model.state_dict(), save_path)