# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from lerobot_policy_diffusion_eecs467.modeling_diffusion_eecs467 import DiffusionEECS467Config, DiffusionEECS467Policy
from lerobot_policy_diffusion_eecs467.processor_diffusion_eecs467 import make_diffusion_eecs467_pre_post_processors

__all__ = [
    "DiffusionEECS467Config",
    "DiffusionEECS467Policy",
    "make_diffusion_eecs467_pre_post_processors",
]