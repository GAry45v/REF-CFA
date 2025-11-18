from typing import Any, List
import torch
import numpy as np
import os

class Reconstruction:
    def __init__(self, unet, config) -> None:
        self.unet = unet
        self.config = config

    def _compute_alpha(self, t):
        """计算 alpha_t_bar 累积乘积"""
        betas = np.linspace(self.config.model.beta_start, self.config.model.beta_end, self.config.model.trajectory_steps, dtype=np.float64)
        betas = torch.tensor(betas, dtype=torch.float, device=self.config.model.device)
        beta = torch.cat([torch.zeros(1, device=self.config.model.device), betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def __call__(self, x: torch.Tensor, y0: torch.Tensor, w: float) -> Any:
        """
        执行完整的去噪重建过程。

        返回:
            - final_reconstruction (torch.Tensor): 最终生成的清晰图像。
            - images_before (List[torch.Tensor]): 包含每一步去噪前的图像列表。
            - images_after (List[torch.Tensor]): 包含每一步去噪后的图像列表。
            - diff_maps (List[torch.Tensor]): 包含每一步前后差异 (xt_next - xt_current) 的图像列表。
            - xs (List[torch.Tensor]): 包含从初始噪声到最终结果的所有中间步骤图像。
            - et_hats (List[torch.Tensor]): (新) 包含每一步的引导后噪声 (epsilon_hat)。
        """
        test_trajectoy_steps = torch.tensor([self.config.model.test_trajectoy_steps], dtype=torch.int64, device=self.config.model.device)
        at = self._compute_alpha(test_trajectoy_steps)
        xt_start = at.sqrt() * x + (1 - at).sqrt() * torch.randn_like(x, device=self.config.model.device)
        
        seq = range(0, self.config.model.test_trajectoy_steps, self.config.model.skip)
        
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            
            # 初始化所有需要记录的列表
            images_before = []
            images_after = []
            diff_maps = []
            xs = [xt_start.clone()] 
            et_hats = [] # <--- (新) 存储 epsilon_hat
            
            xt_current = xt_start
            
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(self.config.model.device)
                next_t = (torch.ones(n) * j).to(self.config.model.device)
                at = self._compute_alpha(t.long())
                at_next = self._compute_alpha(next_t.long())
                
                # 记录去噪前的状态
                images_before.append(xt_current.clone())

                et = self.unet(xt_current, t)
                yt = at.sqrt() * y0 + (1 - at).sqrt() * et
                et_hat = et - (1 - at).sqrt() * w * (yt - xt_current) # <--- 这是您要的 'epsilon' (引导后)
                x0_t = (xt_current - et_hat * (1 - at).sqrt()) / at.sqrt()
                
                c1 = self.config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
                
                # 记录去噪后的状态和差异
                images_after.append(xt_next.clone())
                diff_maps.append(xt_next - xt_current)
                et_hats.append(et_hat.clone()) # <--- (新) 保存 et_hat

                # 更新当前状态以进行下一步，并记录到轨迹中
                xt_current = xt_next
                xs.append(xt_current.clone())
        
        final_reconstruction = images_after[-1]
        
        # (新) 返回 6 个值
        return final_reconstruction, images_before, images_after, diff_maps, xs, et_hats