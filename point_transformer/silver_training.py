"""
Using Flash Attention would enable more efficient training of the model by reducing memory usage and speeding up computation.
However this library requires some specific steps and configurations to be installed on Windows, therefore it wasn't used and tested in this project.
"""

import os
import wandb

from point_transformer.core.pt_data import BackscanDataset
from point_transformer.core.model.model import RegressionPointTransformer
from point_transformer.core.training import *
from config.silver_config import SilverConfig

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np

def custom_collate_fn(batch):
    # batch is a list of (data_dict, target) tuples
    coords = []
    targets = []
    batch_indices = []

    # Validate that all samples have sufficient points
    min_required_points = 48  
    for sample, output in batch:
        if len(sample['coord']) < min_required_points:
            # Either pad the sample or skip it
            print(f"Warning: Sample has only {len(sample['coord'])} points")

    for i, (data_dict, target) in enumerate(batch):
        coords.append(data_dict['coord'])
        targets.append(target)
        batch_indices.extend([i] * len(data_dict['coord']))

    # Concatenate all coordinates
    all_coords = torch.cat(coords, dim=0)

    # Create offset
    num_points_per_sample = [len(c) for c in coords]
    offset = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_points_per_sample), 0)))

    # Create batch tensor
    batch_tensor = torch.tensor(batch_indices, dtype=torch.long)

    # Create final data_dict
    data_dict = {
        'coord': all_coords,
        'feat': all_coords,
        'batch': batch_tensor,
        'offset': offset,
        'grid_size': data_dict['grid_size']
    }

    targets = torch.stack(targets)

    return data_dict, targets


def set_random_seed(seed=42):
    # set random seed for reproducibility
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
            # Reproducibility
            "random_seed": config.random_seed,

            # Model parameters
            "model": config.model_type,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "out_num_points": config.out_num_points,
            

            # Training parameters
            "batch_size": config.batch_size,
            "initial_lr": config.initial_lr,
            "min_lr": config.min_lr,
            "num_epochs": config.num_epochs,
            "optimizer": config.optimizer,
            "weight_decay": config.weight_decay,
            "scheduler": config.scheduler,
            "loss_function": config.loss_function,

            # Dataset parameters
            "backscan_num_points": config.backscan_num_points,
            "sampling_method": config.sampling_method,
            "grid_size": config.grid_size,
            "dataset": config.dataset_name,
            "hospitals": config.hospitals,
            "test_size": 0.1,



            "random_state": config.random_seed,

            # Hardware
            "device": str(device),
            "num_workers": config.num_workers
        }
    )


def main():
    # Load configuration
    config = SilverConfig()

    # Set random seed for reproducibility
    set_random_seed(config.random_seed)

    # Create dataset
    base_dataset = BackscanDataset(
        base_dir=config.base_path,
        num_points=config.backscan_num_points,
        sampling_method=config.sampling_method,
        grid_size=config.grid_size,
        task_type=config.task_type
    )

    # === FULL DATASET ===
    train_dataset_indices, test_dataset_indices = \
        base_dataset.hospital_aware_split(test_size=0.1, random_state=42)

    # create training dataset with augmentation enabled
    full_train_dataset = base_dataset.create_subset(train_dataset_indices)
    # create test dataset with augmentation disabled
    full_test_dataset = base_dataset.create_subset(test_dataset_indices)

    # === SILVER DATASET ===
    silver_train_indices, silver_test_indices = \
        base_dataset.hospital_aware_split(test_size=0.1, random_state=config.random_seed,
                                        hospital_subset=config.hospitals)

    silver_train_dataset = base_dataset.create_subset(silver_train_indices)
    silver_test_dataset = base_dataset.create_subset(silver_test_indices)

    # validate the split
    base_dataset.validate_split(
        train_indices=silver_train_indices,
        test_indices=silver_test_indices
    )

    # Dataloaders (silver dataset)
    silver_train_dataloader = DataLoader(
        silver_train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )
    silver_test_dataloader = DataLoader(
        silver_test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    # initializing the model
    model_dict = {
        'pt_regression': RegressionPointTransformer(
            in_channels=config.in_channels,
            num_points=config.out_num_points,
            output_channels=config.out_channels,
        )
    }

    model = model_dict[config.model_type]

    # optimizer 
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=1e-4
    )

    # loss function 
    loss_function_dict = {
        'mse': MSELoss(),
        'mae': MAELoss(),
        'smooth_l1': SmoothL1Loss(),
    }
    criterion = loss_function_dict.get(config.loss_function, SmoothL1Loss())

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=config.min_lr
    )

    # early stopping configuration
    early_stopping = EarlyStopping(
        patience=config.patience,  # number of epochs to wait before stopping
        min_delta=0,  # minimum change in loss to consider as an improvement
        mode='min',  # we want to minimize the validation loss
        restore_best_weights=True,  # restore the best model weights after stopping
    )

    # mixed precision
    scaler = torch.amp.GradScaler()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)


    # Setup wandb
    setup_wandb(config, device)

    model.to(device)

    best_val_mse = float('inf')
    best_model_path = os.path.join(config.checkpoint_dir, config.best_model_name)

    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        print('-' * 50)

        # training phase
        train_loss, train_metrics = train_epoch_isl(
            model=model,
            dataloader=silver_train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_every_n_batches=config.log_every_n_batches,
            scaler=scaler
        )

        # validation phase
        val_loss, val_metrics = validate_epoch_isl(
            model=model,
            dataloader=silver_test_dataloader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        # update the learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # logging to wandb
        wandb.log({
            'epoch_completed': epoch + 1,
            'epoch_train_loss': train_loss,
            'epoch_train_mse': train_metrics['mse'],
            'epoch_train_mae': train_metrics['mae'],
            'epoch_val_loss': val_loss,
            'epoch_val_mse': val_metrics['mse'],
            'epoch_val_mae': val_metrics['mae'],
            'epoch_learning_rate': current_lr
        })


        # save the best model based on validation MSE
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mse': best_val_mse,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, best_model_path)
            print(f'Saved new best model with MSE: {best_val_mse:.4f} at epoch {epoch + 1}')

        # early stopping
        if early_stopping(val_metrics['mse'], model, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Silver training completed!")
        print(f"Best validation MSE: {best_val_mse:.4f}")

if __name__ == "__main__":
    main()