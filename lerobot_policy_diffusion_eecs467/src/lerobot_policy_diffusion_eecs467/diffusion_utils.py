import math
import torch
import torch.nn as nn
import torchvision
from typing import Callable

class DiffusionNoiseScheduler():
    def __init__(
        self,
        num_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: str = "cuda"
    ):
        self.num_steps = num_steps
        self.device = device
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._get_squaredcos_cap_v2_betas(num_steps, beta_start, beta_end)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        self.betas.to(self.device)

        self.alphas = 1.0 - self.betas
        self.alphas.to(self.device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.timesteps = torch.arange(start=0, end=num_steps).to(self.device)
    
    def _get_squaredcos_cap_v2_betas(self, num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_steps):
            t1 = i / num_steps
            t2 = (i + 1) / num_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), beta_end))
        return torch.tensor(betas, dtype=torch.float32)
    
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_0 = initial input = (B, ...)
            noise = noise to add = (B, ...)
            timesteps = t to propagate noise to (B, )
        """
        alpha_hat_batch = self.alpha_bars[timesteps][:, None]
        x_0_size = x_0.size()

        x_t = torch.sqrt(alpha_hat_batch).view(x_0_size[0], *[1 for _ in range(len(x_0_size) - 1)]) * x_0 + \
            torch.sqrt(1 - alpha_hat_batch).view(x_0_size[0], *[1 for _ in range(len(x_0_size) - 1)]) * noise
        return x_t

    def denoise(
            self,
            x_t: torch.Tensor,
            pred_noise: torch.Tensor,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
        # compute alphas, betas
        alpha_prod_t = self.alpha_bars[timestep]
        alpha_prod_t = alpha_prod_t.to(device=pred_noise.device, dtype=pred_noise.dtype)

        alpha_prod_t_prev = self.alpha_bars[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        alpha_prod_t_prev = alpha_prod_t_prev.to(device=pred_noise.device, dtype=pred_noise.dtype)

        scalar_batch_size = x_t.size(0), *[1 for _ in range(len(x_t.size()) - 1)]
        alpha_prod_t = alpha_prod_t.view(scalar_batch_size)
        alpha_prod_t_prev = alpha_prod_t_prev.view(scalar_batch_size)
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (1.0 / torch.sqrt(alpha_prod_t)) * (x_t - torch.sqrt(1.0 - alpha_prod_t)*pred_noise)
        current_sample_coeff = torch.sqrt(current_alpha_t) * (1.0 - alpha_prod_t_prev) * (1.0 / (1.0 - alpha_prod_t))
        pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev) * current_beta_t * (1.0 / ( 1.0 -  alpha_prod_t) )
   
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t
        variance = 0
        if timestep > 0:
            variance_noise = torch.randn_like(pred_noise)
            
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
            variance = (variance ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
    

# Get the resnet - the image encoder for the diffusion policy model
def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module
