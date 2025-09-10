import os
import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from convolutional_neural_network.isl.config.transfer_config import TransferConfig
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
    wandb.init(
        project=config.wandb_project,
        dir='C:/Users/agmarko/temp/wandb',
        settings=wandb.Settings(
            disable_code=True,
            disable_git=True,
            code_dir=None,
            program_relpath=None,
            git_root=None,
        ),
        config={
            # Transfer learning specific
            "transfer_strategy": config.transfer_strategy,
            "pretrained_model": config.pretrained_model_path,
            "freeze_bn": config.freeze_bn,

            # Model parameters
            "model": config.model_type,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
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
        masks_dir=config.isl_mask_path,  
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

    print(f"Gold Train Dataset Size: {len(gold_train_dataset)}")
    print(f"Gold Test Dataset Size: {len(gold_test_dataset)}")

    # Create dataloaders
    gold_train_dataloader = DataLoader(
        gold_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
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
    model, pretrained_checkpoint = setup_transfer_learning(model, config, device, esl_transfer=False)
    model.to(device)

    # Create optimizer
    optimizer = create_transfer_optimizer(model, config)

    # Loss function - using ISL specific losses
    loss_function_dict = {
        'masked_mse': MaskedMSELoss(),
        'mse': MSELoss(),
        'mae': MAELoss(),
        'smoothl1': SmoothL1Loss()
    }
    criterion = loss_function_dict.get(config.loss_function, MaskedMSELoss())

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.min_lr
    )

    # Early stopping - using rmse_line for ISL
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode='min',  # minimize rmse_line for ISL
        restore_best_weights=True,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler()

    # Create output directory for validation samples
    dataset_type = 'gold'  
    project_root = Path(__file__).resolve().parent
    isl_training_2d_mask_dir = project_root / 'isl_results' / 'transfer' / f"{config.transfer_strategy}" / f"isl_training_2d_mask_{dataset_type}" / config.loss_function
    isl_training_2d_mask_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL training 2d mask results to: {isl_training_2d_mask_dir}")

    # Training loop
    best_val_rmse_line = float('inf')  
    best_model_path = os.path.join(config.checkpoint_dir, config.best_model_name)

    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        print('-' * 50)

        # Handle progressive unfreezing
        optimizer = handle_progressive_unfreezing(model, config, epoch, optimizer)

        # Training phase 
        train_loss, train_metrics = train_epoch_isl(
            model, 
            gold_train_dataloader, 
            criterion, 
            optimizer, 
            device, 
            epoch,
            log_every_n_batches=config.log_every_n_batches, 
            scaler=scaler
        )

        # Validation phase 
        val_loss, val_metrics = validate_epoch_isl(
            model, 
            gold_test_dataloader, 
            criterion, 
            device, 
            epoch
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log to wandb 
        wandb.log({
            'epoch_completed': epoch + 1,
            'epoch_train_loss': train_loss,
            'epoch_train_rmse_line': train_metrics['rmse_line'],
            'epoch_train_mae_line': train_metrics['mae_line'],
            'epoch_train_mae_bg': train_metrics['mae_bg'],
            'epoch_val_loss': val_loss,
            'epoch_val_rmse_line': val_metrics['rmse_line'],
            'epoch_val_mae_line': val_metrics['mae_line'],
            'epoch_val_mae_bg': val_metrics['mae_bg'],
            'epoch_learning_rate': current_lr,
        })

        # Save best model based on validation rmse_line
        if val_metrics['rmse_line'] < best_val_rmse_line:
            best_val_rmse_line = val_metrics['rmse_line']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_rmse_line': best_val_rmse_line,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'transfer_strategy': config.transfer_strategy,
                'pretrained_model_path': config.pretrained_model_path,
            }, best_model_path)
            print(f'Saved new best model with val_rmse_line: {best_val_rmse_line:.4f} at epoch {epoch}')


        if (epoch + 1) % config.visual_inspection_freq == 0:
            # save validation samples for visual inspection 
            save_isl_validation_samples(
                model=model,
                dataloader=gold_test_dataloader,
                device=device,
                epoch=epoch,
                save_dir=isl_training_2d_mask_dir,
                num_samples=5
            )

        # Early stopping check
        if early_stopping(val_metrics['rmse_line'], model, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"ISL transfer learning completed!")
    print(f"Best validation rmse_line: {best_val_rmse_line:.4f}")
    
    # Final cleanup
    wandb.finish()


if __name__ == "__main__":
    main()