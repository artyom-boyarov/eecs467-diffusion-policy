from collections import deque
from .cond_1dunet import ConditionalUnet1D
import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
    ACTION,
    OBS_IMAGES,
)
from lerobot.policies.utils import populate_queues

from lerobot_policy_diffusion_eecs467.diffusion_utils import DiffusionNoiseScheduler, get_resnet
from lerobot_policy_diffusion_eecs467.configuration_diffusion_eecs467 import DiffusionEECS467Config

OBS_IMAGES_TOP = "observation.images.top"

from typing import Dict, Any

class DiffusionEECS467Model(nn.Module):
    def __init__(self, config: DiffusionEECS467Config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.vision_encoder = get_resnet("resnet18")

        self.vision_feature_dim = 512  # ResNet18 output feature dimension
        self.obs_dim = self.vision_feature_dim + self.config.joint_dim
        self.action_dim = self.config.action_dim

        self.noise_pred_unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.config.n_obs_steps,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8
        )

        self.nets = torch.nn.ModuleDict({
            "vision_encoder": self.vision_encoder,
            "noise_pred_unet": self.noise_pred_unet
        })

        self.noise_scheduler = DiffusionNoiseScheduler(
            num_steps=self.config.diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule=self.config.diffusion_beta_schedule,
            device=self.device
        )
        self.nets.to(self.device)
    
    def _concatenate_obs_cond(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, n_obs = batch[OBS_IMAGES_TOP].shape[0], batch[OBS_IMAGES_TOP].shape[1]
        image_features = []
        for image_minibatch in batch[OBS_IMAGES_TOP]:
            image_minibatch_encoded = []
            for i in range(self.config.n_obs_steps):
                image_minibatch_encoded.append(self.vision_encoder(image_minibatch[i].unsqueeze(0)).squeeze())
            image_minibatch_encoded = torch.stack(image_minibatch_encoded)
            image_features.append(image_minibatch_encoded) # (n_obs_steps, vision_feature_dim)

        image_features = torch.stack(image_features) # (B, n_obs_steps, vision_feature_dim)
        state_features = batch[OBS_STATE]
        obs = torch.cat([image_features, state_features], dim=-1) # (B, n_obs_steps * (vision_feature_dim + config.joint_dim))
        return obs


    def generate_actions(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_cond = self._concatenate_obs_cond(inputs)
        batch_size = obs_cond.shape[0]
        obs_cond = obs_cond.flatten(start_dim=1) # (B, n_obs_steps * obs_dim)

        # Start from pure noise
        sample = torch.randn(batch_size, self.config.horizon, self.action_dim).to(obs_cond.device)

        for t in reversed(range(self.config.diffusion_steps)):
            timesteps = torch.full((batch_size,), t, device=obs_cond.device, dtype=torch.long)
            noise_pred = self.noise_pred_unet(sample=sample, timesteps=timesteps, global_cond=obs_cond)
            sample = self.noise_scheduler.denoise(sample, noise_pred, timesteps)

        return sample

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images.top": (B, n_obs_steps, C, H, W)
            "action": (B, action_horizon, action_dim)
        }
        """

        image = batch[OBS_IMAGES_TOP]
        state = batch[OBS_STATE]
        action  = batch[ACTION]

        noise = torch.randn_like(action)
        timesteps = torch.randint(0, self.config.diffusion_steps, (action.shape[0],), device=action.device)
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)
        
        obs_cond = self._concatenate_obs_cond(batch)


        noise_pred = self.noise_pred_unet(sample=noisy_actions, timesteps=timesteps, global_cond=obs_cond.flatten(start_dim=1))
        loss = nn.MSELoss()(noise_pred, noise)
        return loss


class DiffusionEECS467Policy(PreTrainedPolicy):
    """
    Diffusion Policy from "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionEECS467Config
    name = "diffusion_eecs467"

    def __init__(self, config: DiffusionEECS467Config, dataset_stats: Dict[str, Any] | None = None, **kwargs):
        super().__init__(config, dataset_stats, **kwargs)

        config.validate_features()
        self.config = config

        self._queues = None

        self.dp_model = DiffusionEECS467Model(config)

        self.reset()
        
    
    def reset(self):
        self._queues = {
            OBS_IMAGES_TOP: deque(maxlen=self.config.n_obs_steps),
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps)
        }

    def get_optim_params(self):
        return self.dp_model.parameters()
        
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {
            key: torch.stack(list(self._queues[key]), dim=1)
            for key in batch
            if key in self._queues
        }
        return self.dp_model.generate_actions(inputs)
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        
        print("Predicted actions:", self._queues[ACTION])

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss = self.dp_model.loss(batch)
        return loss, None
