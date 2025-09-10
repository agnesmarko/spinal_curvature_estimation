from .base_config import BaseConfig
from dataclasses import dataclass, field
from typing import List

@dataclass
class TransferConfig(BaseConfig):
    # Transfer learning specific
    pretrained_model_path: str = r"path/to/pretrained/model"
    transfer_strategy: str = 'freeze_encoder'  # 'freeze_encoder', 'freeze_decoder', 'freeze_early_layers', 'train_all', 'progressive'
    freeze_bn: bool = True

    # Dataset specific
    # Which hospital's data to use for training and validation in the fine-tuning phase (gold training)
    hospitals: List[str] = field(default_factory=lambda: ['geneva', 'italian','ukbb', 'balgrist'])
    dataset_name: str = 'gold'
    augmentation: bool = True
    test_size: float = 0.1

    # Modified training parameters for fine-tuning
    num_epochs: int = 200
    initial_lr: float = 1e-6
    min_lr: float = 1e-7
    patience: int = 20

    # Multi-LR settings
    encoder_lr_multiplier: float = 0.1
    decoder_lr_multiplier: float = 1.0

    # Progressive unfreezing
    unfreeze_epoch: int = 50
    unfreeze_lr_multiplier: float = 0.1

    # Wandb project
    wandb_project: str = "name_of_wandb_project"

    # Model saving
    checkpoint_dir: str = "path/to/checkpoint/folder"
    best_model_name: str = "best_transfer_model.pth"
