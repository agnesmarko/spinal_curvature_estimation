from .base_config import BaseConfig
from dataclasses import dataclass, field
from typing import List

@dataclass
class SilverConfig(BaseConfig):
    # Dataset specific
    # Which hospital's data to use for training and validation in the pre-training phase (silver training)
    hospitals: List[str] = field(default_factory=lambda: ['croatian', 'italian', 'geneva'])
    dataset_name: str = 'silver'

    # Wandb project
    wandb_project: str = "name_of_wandb_project"

    # Model saving
    checkpoint_dir: str = "path/to/checkpoint/folder"
    best_model_name: str = "best_silver_model.pth"
