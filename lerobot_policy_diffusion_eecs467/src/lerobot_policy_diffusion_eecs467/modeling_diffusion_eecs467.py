from collections import deque
from .cond_1dunet import ConditionalUnet1D
import numpy as np
import torch
import torch.nn as nn
import torchvision

from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
    ACTION,
    OBS_IMAGES,
)
from lerobot.policies.utils import populate_queues, get_output_shape

from lerobot_policy_diffusion_eecs467.diffusion_utils import DiffusionNoiseScheduler, get_resnet
from lerobot_policy_diffusion_eecs467.configuration_diffusion_eecs467 import DiffusionEECS467Config

OBS_IMAGES_TOP = "observation.images.top"

from typing import Dict, Any

# Taken from LeRobot diffusion implementation
class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp = None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = torch.nn.functional.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionEECS467Model(nn.Module):
    def __init__(self, config: DiffusionEECS467Config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.vision_encoder = DiffusionEECS467VisionEncoder(config)

        self.vision_feature_dim = self.vision_encoder.feature_dim  # ResNet18 output feature dimension
        self.obs_dim = self.vision_feature_dim + self.config.joint_dim
        self.action_dim = self.config.action_dim

        self.noise_pred_unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.config.n_obs_steps,
            diffusion_step_embed_dim=self.config.diffusion_step_embed_dim,
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
        image_features = self.nets["vision_encoder"](batch[OBS_IMAGES_TOP])
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

        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = sample[:, start:end]

        return actions

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


class DiffusionEECS467VisionEncoder(nn.Module):
    def __init__(self, config: DiffusionEECS467Config):
        super().__init__()
        self.config = config
        self.resnet = get_resnet("resnet18", weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) # Remove avgpool and fc layers
        
        self.feature_map_shape = get_output_shape(self.resnet, (1, 3, *config.crop_shape))[1:]
        self.pool = SpatialSoftmax(input_shape=self.feature_map_shape, num_kp=config.spatial_softmax_kp)
        self.feature_dim = config.spatial_softmax_kp * 2
        self.linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

        self.encode_image = nn.Sequential(
            self.resnet,
            self.pool,
            nn.Flatten(),
            self.linear,
            self.relu
        )
        self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
        self.random_crop = torchvision.transforms.RandomCrop(config.crop_shape) if config.random_crop else self.center_crop

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:

        # Center/Random crop for training/inference.
        if self.training:
            image_batch = self.random_crop(image_batch.flatten(0, 1)) # (B * n_obs_steps, C, H, W)
        else:
            image_batch = self.center_crop(image_batch.flatten(0, 1)) # (B * n_obs_steps, C, H, W)
        
        image_batch = image_batch.view(-1, self.config.n_obs_steps, *image_batch.shape[1:]) # (B, n_obs_steps, C, H, W)
        
        image_features = []
        for image_minibatch in image_batch:
            image_minibatch_encoded = []
            for i in range(self.config.n_obs_steps):
                image_minibatch_encoded.append(self.encode_image(image_minibatch[i].unsqueeze(0)).squeeze())
            image_minibatch_encoded = torch.stack(image_minibatch_encoded)
            image_features.append(image_minibatch_encoded) # (n_obs_steps, vision_feature_dim)
        return torch.stack(image_features) # (B, n_obs_steps, vision_feature_dim)

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
        print("State inputs:", inputs[OBS_STATE])
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
