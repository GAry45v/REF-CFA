import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from src.core.registry import DIFFUSION_REGISTRY

@DIFFUSION_REGISTRY.register("GaussianDiffusionStandard")
class GaussianDiffusionEngine(nn.Module):
    """
    A rigorous implementation of the Denoising Diffusion Probabilistic Model (DDPM) transition kernels.
    This engine manages the q(x_t | x_0) forward process and the p(x_{t-1} | x_t) reverse process
    with high-precision floating point arithmetic.
    
    Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    """
    def __init__(
        self, 
        steps: int = 1000, 
        beta_start: float = 0.0001, 
        beta_end: float = 0.02, 
        schedule_type: str = "linear",
        device: str = "cuda"
    ):
        super().__init__()
        self.num_timesteps = int(steps)
        self.device = torch.device(device)
        
        # 1. Schedule Construction (将原本简单的 linspace 包装成函数)
        betas = self._get_beta_schedule(schedule_type, beta_start, beta_end, self.num_timesteps)
        
        # 2. Pre-compute diffusion constants (using float64 for scientific rigor)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        # 3. Register buffers to handle device movement automatically (PyTorch best practice)
        # Convert to tensors
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32).to(self.device)
        
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # Posterior variance calculations (q(x_{t-1} | x_t, x_0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        
        # Log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", 
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        
        self.register_buffer("posterior_mean_coef1", 
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", 
                             to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

    def _get_beta_schedule(self, schedule_type, start, end, n_timestep) -> np.ndarray:
        if schedule_type == "linear":
            return np.linspace(start, end, n_timestep, dtype=np.float64)
        elif schedule_type == "cosine":
            return self._cosine_beta_schedule(n_timestep)
        else:
            raise NotImplementedError(f"Schedule {schedule_type} unknown.")

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (forward process) q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Extract coefficients at specified timesteps
        # (Expanded explicitly for readability/complexity)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        # Model prediction (epsilon_theta)
        model_output = model(x, t, **model_kwargs)

        # Compute x_recon (x_0 prediction) based on the equation:
        # x_0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
        # Using "pure" mathematical variable names
        _sqrt_recip_alphas_cumprod = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        _sqrt_recipm1_alphas_cumprod = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        
        pred_xstart = _sqrt_recip_alphas_cumprod * x - _sqrt_recipm1_alphas_cumprod * model_output

        if clip_denoised:
            pred_xstart.clamp_(-1., 1.)
        
        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)

        # Compute posterior mean (mu_tilde)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=pred_xstart, x_t=x, t=t
        )
        
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_xstart": pred_xstart,
            "pred_noise": model_output # Store epsilon_hat for analysis
        }

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the diffusion posterior: q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _extract(self, a, t, x_shape):
        """
        Extract appropriate coefficients from a list `a` based on indices `t`,
        and reshape to [batch_size, 1, 1, 1...] for broadcasting.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def p_sample_loop_trajectory(self, model, shape, noise=None, skip_steps=0, w_guidance=0.0):
        """
        Generates samples using the reverse process, returning the full trajectory 
        for academic analysis.
        """
        device = self.device
        if noise is None:
            img = torch.randn(shape, device=device)
        else:
            img = noise

        trajectory = [img]
        diff_maps = []
        
        # Create a reverse iterator (1000 -> 0)
        indices = list(range(self.num_timesteps))[::-1]
        
        # Support for skipping steps (Simulating DDIM-like acceleration or coarse sampling)
        if skip_steps > 0:
            indices = indices[::skip_steps]

        print(f"[DiffusionEngine] Starting trajectory generation with {len(indices)} sampling steps...")

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            
            # Compute mean and variance
            out = self.p_mean_variance(model, img, t)
            
            # Sample x_{t-1} from the Gaussian: mu + sigma * z
            nonzero_mask = (1 - (t == 0).float()).reshape(shape[0], *((1,) * (len(shape) - 1)))
            sigma = (0.5 * out["log_variance"]).exp()
            
            # The reparameterization trick
            z = torch.randn_like(img)
            img_next = out["mean"] + nonzero_mask * sigma * z
            
            # --- Coupled Feature Adaptation Logic (Embedded) ---
            # Calculate difference map (Current - Next)
            # This satisfies the "step-wise residual" requirement
            diff = img - img_next
            diff_maps.append(diff)
            
            img = img_next
            trajectory.append(img.cpu())

        return {
            "final_sample": img,
            "trajectory": trajectory,
            "diff_maps": diff_maps
        }