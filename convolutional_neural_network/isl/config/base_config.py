import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BaseConfig:
    # Data paths
    base_path: str = r'path\to\the\base\folder'
    depthmap_path: str = os.path.join(base_path, 'backscan')
    isl_mask_path: str = os.path.join(base_path, 'isl_masks')

    # Dataset parameters
    # Which hospital's data to augment during training
    augment_hospitals: List[str] = field(default_factory=lambda: ['ukbb', 'balgrist', 'croatian', 'italian', 'geneva'])

    # Model parameters
    model_type: str = 'unet'  # 'unet' or 'r2attunet'
    in_channels: int = 1
    out_channels: int = 1
    start_channels: int = 56
    depth: int = 5
    mode: str = 'transpose'  # 'transpose' or 'bilinear'

    # Training parameters
    batch_size: int = 16
    num_epochs: int = 600
    initial_lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'  # the model was tested with this optimizer only
    scheduler: str = "CosineAnnealingLR" # the model was tested with this scheduler only

    # Loss function
    # 'masked_mse', 'mse', 'mae', 'smoothl1' in the case of ISL segmentation (regression)
    loss_function: str = 'mse'  

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001

    # Reproducibility
    random_seed: int = 42

    # Hardware
    num_workers: int = 4

    # Logging - wandb
    log_every_n_batches: int = 5

    # Visual inspection
    # How frequently to save generated output images during training (measured in epochs)
    visual_inspection_freq: int = 1

