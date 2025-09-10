
import os
import wandb

from convolutional_neural_network.core.cnn_data import DepthmapDataset
from convolutional_neural_network.core.model import UNet, R2AttUNet
from convolutional_neural_network.core.training import *
from convolutional_neural_network.isl.config.silver_config import SilverConfig

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import numpy as np


def set_random_seed(seed=42):
    # set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

 
def diagnose_dataset(dataset, name, num_samples=10):
    print(f"\n=== {name} Dataset Diagnosis ===")
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image: shape={image.shape}, min={image.min():.3f}, max={image.max():.3f}")
        print(f"  Mask: shape={mask.shape}, min={mask.min():.3f}, max={mask.max():.3f}, non-zero={mask.sum():.3f}")

        # check if mask is completely empty
        if mask.sum() == 0:
            print(f"  WARNING: Mask {i} is completely empty!")


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
            "start_channels": config.start_channels,
            "depth": config.depth,
            "mode": config.mode,

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
            "dataset": config.dataset_name,
            "hospitals": config.hospitals,
            "augmentation": True,
            "test_size": 0.1,
            "random_state": config.random_seed,

            # Hardware
            "device": str(device),
            "num_workers": config.num_workers
        }
    )


def main():

    # Initialize config
    config = SilverConfig()

    # Set random seed
    set_random_seed(config.random_seed)

    # Create base dataset
    base_dataset = DepthmapDataset(
        images_dir=config.depthmap_path,
        masks_dir=config.isl_mask_path,
        augment_hospitals=config.augment_hospitals,
    )

    # === FULL DATASET ===
    train_dataset_indices, test_dataset_indices = \
        base_dataset.hospital_aware_split(test_size=0.1, random_state=42)

    # create training dataset with augmentation enabled
    full_train_dataset = base_dataset.create_subset(train_dataset_indices, is_training=True)
    # create test dataset with augmentation disabled
    full_test_dataset = base_dataset.create_subset(test_dataset_indices, is_training=False)

    # === SILVER DATASET ===
    silver_train_indices, silver_test_indices = \
        base_dataset.hospital_aware_split(test_size=0.1, random_state=config.random_seed,
                                        hospital_subset=config.hospitals)

    silver_train_dataset = base_dataset.create_subset(silver_train_indices, is_training=True)
    silver_test_dataset = base_dataset.create_subset(silver_test_indices, is_training=False)

    # validate the split
    base_dataset.validate_split(
        train_indices=silver_train_indices,
        test_indices=silver_test_indices
    )

    # test the datasets
    diagnose_dataset(silver_train_dataset, "Silver Train")
    diagnose_dataset(silver_test_dataset, "Silver Test")

    # print out the sizes of the datasets
    print(f"Silver Train Dataset Size: {len(silver_train_dataset)}")
    print(f"Silver Test Dataset Size: {len(silver_test_dataset)}")


    # creating dataloaders

    # silver dataset dataloaders
    silver_train_dataloader = DataLoader(
        silver_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )       

    silver_test_dataloader = DataLoader(
        silver_test_dataset,        
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )



    # not shuffling the test dataset is considered good practice:
    # it ensures that the evaluation is consistent and reproducible across different runs


    # initializing the model
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


    # optimizer 
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=1e-4
    )

    # loss function 
    loss_function_dict = {
        'masked_mse': MaskedMSELoss(),
        'mse': MSELoss(),
        'mae': MAELoss(),
        'smoothl1': SmoothL1Loss()
    }
    criterion = loss_function_dict.get(config.loss_function, MSELoss())


    # scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.min_lr
    )

    # early stopping configuration
    early_stopping = EarlyStopping(
        patience=config.patience,  # number of epochs to wait before stopping
        min_delta=0,  # minimum change in rmse_line to consider as an improvement
        mode='min',  # we want to minimize the validation rmse_line
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

    best_val_rmse_line = float('inf')
    best_model_path = os.path.join(config.checkpoint_dir, config.best_model_name)

    # Create output directory
    dataset_type = 'silver'
    project_root = Path(__file__).resolve().parent
    isl_training_2d_mask_dir = project_root / 'isl_results' / 'silver' / f"isl_training_2d_mask_only_pred_{dataset_type}" / config.loss_function
    isl_training_2d_mask_dir.mkdir(parents=True, exist_ok=True)

    isl_training_2d_mask_dir2 = project_root / 'isl_results' / 'silver' / f"isl_training_2d_mask_{dataset_type}" / config.loss_function
    isl_training_2d_mask_dir2.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL training 2d mask results to: {isl_training_2d_mask_dir}")
    print(f"Saving ISL training 2d mask results to: {isl_training_2d_mask_dir2}")

    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        print('-' * 50)

        # training phase
        train_loss, train_metrics = train_epoch_isl(
            model,
            silver_train_dataloader,
            criterion,
            optimizer,
            device,
            epoch,
            log_every_n_batches=5,
            scaler=scaler
        )

        # validation phase
        val_loss, val_metrics = validate_epoch_isl(
            model,
            silver_test_dataloader,
            criterion,
            device,
            epoch
        )

        # update the learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]


        # logging to wandb
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
            'epoch_learning_rate': current_lr

        })

        # save the best model based on validation rmse_line
        if val_metrics['rmse_line'] < best_val_rmse_line:
            best_val_rmse_line = val_metrics['rmse_line']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_rmse_line': best_val_rmse_line,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, best_model_path)
            print(f'Saved new best model with val_rmse_line: {best_val_rmse_line:.4f} at epoch {epoch}')


        if (epoch + 1) % config.visual_inspection_freq == 0:
            # save validation samples for visual inspection
            save_isl_validation_samples_only_the_prediction(
                model=model,
                dataloader=silver_test_dataloader,
                device=device,
                epoch=epoch,
                save_dir=isl_training_2d_mask_dir,
                num_samples=5
            )

        if (epoch + 1) % config.visual_inspection_freq == 0:
            # save validation samples for visual inspection
            save_isl_validation_samples(
                model=model,
                dataloader=silver_test_dataloader,
                device=device,
                epoch=epoch,
                save_dir=isl_training_2d_mask_dir2,
                num_samples=5
            )

        # Early stopping check
        if early_stopping(val_metrics['rmse_line'], model, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Silver training completed!")
        print(f"Best validation rmse_line: {best_val_rmse_line:.4f}")
        print(f"Epoch validation rmse_line: {val_metrics['rmse_line']:.4f}")


if __name__ == "__main__":
    main()

