
from dataclasses import dataclass, field
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode

@PreTrainedConfig.register_subclass("diffusion_eecs467")
@dataclass
class DiffusionEECS467Config(PreTrainedConfig):

    # Dimensions
    action_dim = 6
    joint_dim = 6

    # Horizons
    n_obs_steps: int = 8
    n_action_steps: int = 32
    horizon: int = 64

    # Diffusion model parameters
    diffusion_steps: int = 100
    diffusion_beta_schedule: str = "squaredcos_cap_v2"

    # Training parameters
    learning_rate: float = 1e-4
    betas: tuple = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(#
        default_factory=lambda: {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    })



    def __post_init__(self):
        return super().__post_init__()

    def validate_features(self):
        pass

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.learning_rate, # Learning rate
            betas=self.betas, # Used to compute running averages of gradient and its square
            eps=self.eps, # Term added to the denominator to improve numerical stability
            weight_decay=self.weight_decay # L2 regularization penalty coefficient
        )
    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )
    
    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
