import os
import sys
import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from convolutional_neural_network.esl.config.transfer_config import TransferConfig
from convolutional_neural_network.core.cnn_data import DepthmapDataset
from convolutional_neural_network.core.model import UNet, R2AttUNet
from convolutional_neural_network.core.training import *
from convolutional_neural_network.core.transfer_utils import setup_transfer_learning, create_transfer_optimizer, handle_progressive_unfreezing

def set_random_seed(seed=42):
    # set random seed for reproducibility
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_wandb(config, device):
    # Set wandb environment variables
    wandb.init(
        project=config.wandb_project,
        dir='C:/Users/agmarko/temp/wandb',
        config={
            # Transfer learning specific
            "transfer_strategy": config.transfer_strategy,
            "pretrained_model": config.pretrained_model_path,
            "freeze_bn": config.freeze_bn,

            # Model parameters
            "model": config.model_type,
            "in_channels": 1,
            "out_channels": 1,
            "start_channels": config.start_channels,
            "depth": config.depth,
            "mode": config.mode,

            # Training parameters
            "batch_size": config.batch_size,
            "initial_lr": config.initial_lr,
            "min_lr": config.min_lr,
            "num_epochs": config.num_epochs,
            "loss_function": config.loss_function,
            'scheduler': config.scheduler,
            "weight_decay": config.weight_decay,

            # Dataset parameters
            "dataset": config.dataset_name,
            "hospitals": config.hospitals,
            "random_seed": config.random_seed,
            "augmentation": config.augmentation,
            "test_size": config.test_size,

             # Hardware
            "device": str(device),
            "num_workers": config.num_workers
        }
    )


def main():
    # Initialize configuration
    config = TransferConfig()

    # Set random seed
    set_random_seed(config.random_seed)

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    

    # Create dataset
    base_dataset = DepthmapDataset(
        images_dir=config.depthmap_path,
        masks_dir=config.esl_mask_path,
        augment_hospitals=config.augment_hospitals,
    )

    # Create gold dataset splits
    gold_train_indices, gold_test_indices = base_dataset.hospital_aware_split(
        test_size=0.1, 
        random_state=config.random_seed,
        hospital_subset=config.hospitals
    )

    gold_train_dataset = base_dataset.create_subset(gold_train_indices, is_training=True)
    gold_test_dataset = base_dataset.create_subset(gold_test_indices, is_training=False)

    # validate the split
    base_dataset.validate_split(
        train_indices=gold_train_indices,
        test_indices=gold_test_indices
    )

    # Create dataloaders
    gold_train_dataloader = DataLoader(
        gold_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    gold_test_dataloader = DataLoader(
        gold_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Initialize model
    model_dict = {
        'unet': UNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            start_channels=config.start_channels,
            depth=config.depth,
            mode=config.mode,
        ),
        'r2attunet': R2AttUNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            start_channels=config.start_channels,
            depth=config.depth,
            t=1,
            mode=config.mode,
            use_attention=True,
            dropout=0.3
        )
    }

    model = model_dict[config.model_type]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Setup wandb
    setup_wandb(config, device)

    # Setup transfer learning
    model, pretrained_checkpoint = setup_transfer_learning(model, config, device)
    model.to(device)

     # Create optimizer
    optimizer = create_transfer_optimizer(model, config)

    # Loss function
    loss_function_dict = {
        'dice': DiceLoss(),
        'focal': FocalLoss(),
        'tversky': TverskyLoss(),
        'dice+focal': DiceFocalLoss(),
    }
    criterion = loss_function_dict.get(config.loss_function, DiceFocalLoss())

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.min_lr
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,  # number of epochs to wait before stopping
        min_delta=config.min_delta, # minimum change in validation dice to consider as an improvement
        mode='max', # we want to maximize the validation dice
        restore_best_weights=True, # restore the best model weights after stopping
    )

    # Mixed precision
    scaler = torch.amp.GradScaler()

     # Training loop
    best_val_dice = 0.0
    best_model_path = os.path.join(config.checkpoint_dir, config.best_model_name)

    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        print('-' * 50)

        # Handle progressive unfreezing
        optimizer = handle_progressive_unfreezing(model, config, epoch, optimizer)


        # Training phase
        train_loss, train_metrics = train_epoch_esl(
            model, gold_train_dataloader, criterion, optimizer, device, epoch,
            log_every_n_batches=config.log_every_n_batches, scaler=scaler
        )

        # Validation phase
        val_loss, val_metrics = validate_epoch_esl(
            model, gold_test_dataloader, criterion, device, epoch, scaler
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_dice': train_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_loss': val_loss,
            'val_dice': val_metrics['dice'],
            'val_iou': val_metrics['iou'],
            'learning_rate': current_lr,
        })

        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
                'config': config,
                'transfer_strategy': config.transfer_strategy,
                'pretrained_model_path': config.pretrained_model_path,
            }, best_model_path)
            print(f'Saved new best model with val_dice: {best_val_dice:.4f}')

        # Early stopping check
        if early_stopping(val_metrics['dice'], model, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"\nTransfer learning completed!")
        print(f"Best validation dice: {best_val_dice:.4f}")
        print(f"Epoch validation dice: {val_metrics['dice']:.4f}")


if __name__ == "__main__":
    main()


