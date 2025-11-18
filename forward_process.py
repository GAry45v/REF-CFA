# forward_process.py
import torch
import numpy as np

def forward_diffusion_visualize(x0, total_steps, display_steps, config):
    """
    生成并返回前向加噪过程中的一系列中间图像。
    这个序列将包含原始图像x0。
    """
    betas = np.linspace(config.model.beta_start, config.model.beta_end, total_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    alphas_cumprod = (1. - b).cumprod(dim=0)
    
    # 我们希望总共展示 N+1 张图（包括x0），所以linspace的点数是 N
    steps_to_show = torch.linspace(0, total_steps - 1, display_steps, dtype=torch.long).to(config.model.device)
    
    # 列表的第一个元素是原始的清晰图像
    noisy_images = [x0.clone()]
    
    with torch.no_grad():
        for t in steps_to_show:
            at = alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
            e = torch.randn_like(x0, device=x0.device)
            xt = at.sqrt() * x0 + (1 - at).sqrt() * e
            noisy_images.append(xt.clone())
            
    return noisy_images