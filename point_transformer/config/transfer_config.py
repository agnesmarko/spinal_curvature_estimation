import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class TransferConfig:
    # Data path
    base_path: str = r'path\to\the\base\folder'

    # Dataset specific
    # Which hospital's data to use for training and validation in the fine-tuning phase (gold training)
    hospitals: List[str] = field(default_factory=lambda: ['ukbb', 'italian', 'geneva', 'balgrist'])
    dataset_name: str = 'gold'
    task_type: str = 'isl_pred'  # 'esl_pred' or 'isl_pred'

    # Reproducibility
    random_seed: int = 42

    # Model saving
    checkpoint_dir: str = "path/to/checkpoint/folder"
    best_model_name: str = "best_transfer_model.pth"