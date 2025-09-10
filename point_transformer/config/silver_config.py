import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class SilverConfig:
    # Data path
    base_path: str = r'P:\Projects\LMB_4Dspine\database\back_scans'

    # Dataset specific
    # Which hospital's data to use for training and validation in the pre-training phase (silver training)
    hospitals: List[str] = field(default_factory=lambda: ['croatian'])
    dataset_name: str = 'silver'
    task_type: str = 'esl_pred'  # 'esl_pred' or 'isl_pred'

    # Wandb project
    wandb_project: str = "name_of_wandb_project"

    # Model parameters
    model_type: str = 'pt_regression' # only this type is supported
    in_channels: int = 3
    out_channels: int = 48
    out_num_points: int = 16
    backscan_num_points: int = 2048 # 1024 or 2048 or 4096 or 8192
    sampling_method: str = 'fps'  # 'fps' or 'random'
    grid_size: float = 0.01

    # Training parameters
    batch_size: int = 16
    num_epochs: int = 600
    initial_lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'   # the model was tested with this optimizer only
    scheduler: str = "CosineAnnealingWarmRestarts" # the model was tested with this scheduler only
    num_workers: int = 4

    # Loss function
    loss_function: str = 'mse'  # 'smooth_l1', 'mse', 'mae'

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001

    # Reproducibility
    random_seed: int = 42

    # Hardware
    num_workers: int = 1

    # Logging - wandb
    log_every_n_batches: int = 5

    # Model saving
    checkpoint_dir: str = "path/to/checkpoint/folder"
    best_model_name: str = "best_silver_model.pth"

