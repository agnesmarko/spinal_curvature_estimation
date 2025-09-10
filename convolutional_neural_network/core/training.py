import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
import random


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth # small constant to avoid division by zero

    def forward(self, pred, target):
        # apply sigmoid activation to predictions
        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # calculate pt = probability of the true class
        # this is a binary classification problem -> bce_loss = -log(pt), therefore:
        pt = torch.exp(-bce_loss)


        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss
    

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        # alfa + beta is 1
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false negatives
        self.beta = beta    # weight for false positives
        self.smooth = smooth

    def forward(self, pred, target):
        # apply sigmoid activation to predictions
        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        return 1 - tversky
    
 
class DiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2)

    def forward(self, pred, target):
        return 0.7 * self.dice(pred, target) + 0.3 * self.focal(pred, target)


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum') # use 'sum' to average over non-zero pixels only

    def forward(self, pred, target):
        # here we only want to calculate loss on the pixels that are part of the line in the ground truth
        # assuming non-line pixels are 0 in the target mask
        mask = (target > 0).float()
        
        # apply the mask to both prediction and target
        pred_masked = pred * mask
        target_masked = target * mask
        
        # calculate loss only on the relevant pixels
        loss = self.mse_loss(pred_masked, target_masked)
        
        # normalize the loss by the number of non-zero pixels
        num_pixels = torch.sum(mask) + 1e-6 # add epsilon to avoid division by zero
        
        return loss / num_pixels
    

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        # standard MSELoss, which will average over all pixels in the tensor.
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # the model's output (pred) has no activation function.
        # the target mask has depth values on the line and 0 for the background.
        # by applying MSE directly, we penalize the model for:
        # 1. incorrect depth values on the line.
        # 2. non-zero predictions in the background.
        return self.mse_loss(pred, target)
    

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        # standard MAELoss, which will average over all pixels in the tensor.
        self.mae_loss = nn.L1Loss()

    def forward(self, pred, target):
        # the model's output (pred) has no activation function.
        # the target mask has depth values on the line and 0 for the background.
        # by applying MAE directly, we penalize the model for:
        # 1. incorrect depth values on the line.
        # 2. non-zero predictions in the background.
        return self.mae_loss(pred, target)


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, pred, target):
        return self.smooth_l1_loss(pred, target)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='max', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode # 'max' for metrics like IoU, 'min' for loss
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1


    def __call__(self, current_score, model, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        
        # check if current score is better than the best score
        if self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(f"Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"Restored weights from epoch {self.best_epoch}")
                return True
            
            return False
    

def calculate_segmentation_metrics(pred, target, threshold=0.5):
    # calculate metrics (SPL segmentation)

    # apply sigmoid activation to predictions
    pred = torch.sigmoid(pred)

    # convert to binary masks
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # calculate metrics using pytorch (slightly more efficient than numpy)
    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection

    # IoU (jaccard index)
    iou = intersection / (union + 1e-6)  # small constant to avoid division by zero

    # Dice score
    dice = (2 * intersection) / (torch.sum(pred_binary) + torch.sum(target_binary) + 1e-6)

     # calculate with adaptive threshold (mean of predictions)
    adaptive_threshold = pred.mean()
    pred_binary_adaptive = (pred > adaptive_threshold).float()
    intersection_adaptive = torch.sum(pred_binary_adaptive * target_binary)
    dice_adaptive = (2 * intersection_adaptive) / (torch.sum(pred_binary_adaptive) + torch.sum(target_binary) + 1e-6)

    # pixel accuracy
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    pixel_accuracy = torch.sum(pred_flat == target_flat) / torch.numel(pred_flat)

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'dice_adaptive': dice_adaptive.item(),
        'pixel_accuracy': pixel_accuracy.item(),
        'pred_mean': pred.mean().item(),
        'pred_std': pred.std().item()
    }
    

def calculate_regression_metrics(pred, target):
    # calculate metrics (ISL segmentation - regression)

    # create masks for line and background
    line_mask = (target > 0).float()
    bg_mask = (target == 0).float()

    # --- Line Metrics ---
    line_pixels = torch.sum(line_mask) + 1e-8
    line_pred = pred * line_mask
    line_target = target * line_mask
    
    line_se = (line_pred - line_target)**2
    line_ae = torch.abs(line_pred - line_target)
    
    rmse_line = torch.sqrt(torch.sum(line_se) / line_pixels)
    mae_line = torch.sum(line_ae) / line_pixels

    # --- Background Metrics ---
    bg_pixels = torch.sum(bg_mask) + 1e-8
    bg_pred = pred * bg_mask
    
    mae_bg = torch.sum(torch.abs(bg_pred)) / bg_pixels

    # --- Overall Metrics ---
    overall_mae = F.l1_loss(pred, target)
    overall_rmse = torch.sqrt(F.mse_loss(pred, target))

    return {
        'rmse_line': rmse_line.item(),
        'mae_line': mae_line.item(),
        'mae_bg': mae_bg.item(),
        'rmse_overall': overall_rmse.item(),
        'mae_overall': overall_mae.item()
    }


def train_epoch_esl(model, dataloader, criterion, optimizer, device, epoch, log_every_n_batches=5, scaler=None, convert_to_binary=False):
    # training function for SPL(ESL) segmentation
    
    # training for one epoch
    model.train()

    # initialize scaler
    if scaler is None:
        scaler = torch.amp.GradScaler()

    running_loss = 0.0
    running_metrics = {'iou': 0.0, 'dice': 0.0, 'dice_adaptive': 0.0, 'pixel_accuracy': 0.0}

    # for batch-level logging
    batch_running_loss = 0.0
    batch_running_metrics = {'iou': 0.0, 'dice': 0.0, 'dice_adaptive': 0.0, 'pixel_accuracy': 0.0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} training')

    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        # convert ISL masks to binary if specified
        if convert_to_binary:
            masks = (masks > 0).float()


        optimizer.zero_grad()

        # with mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        # mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # with mixed precision
        # optimizer step
        scaler.step(optimizer)
        scaler.update()

        # calculate metrics
        with torch.no_grad():
            batch_metrics = calculate_segmentation_metrics(outputs, masks)

        # detailed logging
        if batch_idx % 50 == 0:
            print(f"\nBatch {batch_idx} Details:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Pred stats: mean={outputs.mean():.4f}, std={outputs.std():.4f}")
            print(f"  Target stats: mean={masks.mean():.4f}, std={masks.std():.4f}")
            print(f"  Dice (0.5): {batch_metrics['dice']:.4f}")
            print(f"  Dice (adaptive): {batch_metrics['dice_adaptive']:.4f}")
            print(f"  Adaptive threshold: {batch_metrics['pred_mean']:.4f}")

        # update running statistics
        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += batch_metrics[key]

        # update batch-level running statistics
        batch_running_loss += loss.item()
        for key in batch_running_metrics:
            batch_running_metrics[key] += batch_metrics[key]

        # update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{batch_metrics["iou"]:.4f}',
            'Dice': f'{batch_metrics["dice"]:.4f}',
            'Dice_Adaptive': f'{batch_metrics["dice_adaptive"]:.4f}', 
            'Pred_Mean': f'{batch_metrics["pred_mean"]:.3f}'          
        })


        # log to wandb every N batches
        if (batch_idx + 1) % log_every_n_batches == 0:
            # calculate averages over the last N batches
            avg_batch_loss = batch_running_loss / log_every_n_batches
            avg_batch_dice = batch_running_metrics['dice'] / log_every_n_batches
            avg_batch_dice_adaptive = batch_running_metrics['dice_adaptive'] / log_every_n_batches

            # calculate global step for wandb
            global_step = epoch * len(dataloader) + batch_idx + 1

            # get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # log to wandb
            if (batch_idx + 1) % 50 == 0:
                wandb.log({
                    'step': global_step,
                    'batch_train_loss': avg_batch_loss,
                    'batch_train_dice': avg_batch_dice,
                    'batch_train_dice_adaptive': avg_batch_dice_adaptive,
                    'batch_learning_rate': optimizer.param_groups[0]['lr']
                })

            # reset batch-level running statistics
            batch_running_loss = 0.0
            batch_running_metrics = {'iou': 0.0, 'dice': 0.0, 'dice_adaptive': 0.0, 'pixel_accuracy': 0.0}

    # calculate epoch average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}    

    return epoch_loss, epoch_metrics


def train_epoch_isl(model, dataloader, criterion, optimizer, device, epoch, log_every_n_batches=5, scaler=None):
    # training function for ISL segmentation (regression)

    # training for one epoch for the ISL regression task
    model.train()

    # initialize scaler for mixed precision
    if scaler is None:
        scaler = torch.amp.GradScaler()

    # metrics for the entire epoch
    running_loss = 0.0
    running_metrics = {'rmse_line': 0.0, 'mae_line': 0.0, 'mae_bg': 0.0, 'rmse_overall': 0.0, 'mae_overall': 0.0}

    # metrics for batch-level logging
    batch_running_loss = 0.0
    batch_running_metrics = {key: 0.0 for key in running_metrics}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} training')

    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        # forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        # backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # optimizer step
        scaler.step(optimizer)
        scaler.update()

        # calculate metrics
        with torch.no_grad():
            batch_metrics = calculate_regression_metrics(outputs, masks)

        # detailed logging for debugging
        if batch_idx % 50 == 0:
            print(f"\nBatch {batch_idx} Details:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Pred stats: mean={outputs.mean():.4f}, std={outputs.std():.4f}")
            print(f"  Target stats: mean={masks.mean():.4f}, std={masks.std():.4f}")
            print(f"  RMSE (Line): {batch_metrics['rmse_line']:.4f}")
            print(f"  MAE (BG): {batch_metrics['mae_bg']:.4f}")

        # update running statistics for the epoch
        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += batch_metrics[key]

        # update running statistics for batch logging
        batch_running_loss += loss.item()
        for key in batch_running_metrics:
            batch_running_metrics[key] += batch_metrics[key]

        # update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'RMSE_Line': f'{batch_metrics["rmse_line"]:.4f}',
            'MAE_BG': f'{batch_metrics["mae_bg"]:.4f}'
        })

        # log to wandb every N batches
        if (batch_idx + 1) % log_every_n_batches == 0:
            # calculate averages over the last N batches
            avg_batch_loss = batch_running_loss / log_every_n_batches
            avg_batch_metrics = {key: value / log_every_n_batches for key, value in batch_running_metrics.items()}
            
            # calculate global step for wandb
            global_step = epoch * len(dataloader) + batch_idx + 1

            # log to wandb
            wandb.log({
                'step': global_step,
                'batch_train_loss': avg_batch_loss,
                'batch_train_rmse_line': avg_batch_metrics['rmse_line'],
                'batch_train_mae_line': avg_batch_metrics['mae_line'],
                'batch_train_mae_bg': avg_batch_metrics['mae_bg'],
                'batch_learning_rate': optimizer.param_groups[0]['lr']
            })

            # reset batch-level running statistics
            batch_running_loss = 0.0
            batch_running_metrics = {key: 0.0 for key in running_metrics}

    # calculate epoch average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}

    return epoch_loss, epoch_metrics


def validate_epoch_esl(model, dataloader, criterion, device, epoch, scaler=None, convert_to_binary=False):
    # validation function for SPL(ESL) segmentation

    # validation for one epoch
    model.eval()
    running_loss = 0.0
    running_metrics = {'iou': 0.0, 'dice': 0.0, 'pixel_accuracy': 0.0}

    if scaler is None:
        scaler = torch.amp.GradScaler()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} validation')

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            # convert ISL masks to binary if specified
            if convert_to_binary:
                masks = (masks > 0).float()

            # with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                v_loss = criterion(outputs, masks)

            # calculate metrics
            batch_metrics = calculate_segmentation_metrics(outputs, masks)

            # update running statistics
            running_loss += v_loss.item()
            for key in running_metrics:
                running_metrics[key] += batch_metrics[key]

            # update progress bar
            pbar.set_postfix({
                'Loss': f'{v_loss.item():.4f}',
                'IoU': f'{batch_metrics["iou"]:.4f}',
                'Dice': f'{batch_metrics["dice"]:.4f}'
            })

    # calculate epoch average validation loss and metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}

    return epoch_loss, epoch_metrics


def validate_epoch_isl(model, dataloader, criterion, device, epoch):
    # validation function for ISL segmentation (regression)

    # validation for one epoch for the ISL regression task
    model.eval()
    running_loss = 0.0
    running_metrics = {'rmse_line': 0.0, 'mae_line': 0.0, 'mae_bg': 0.0, 'rmse_overall': 0.0, 'mae_overall': 0.0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} validation')

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            # forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                v_loss = criterion(outputs, masks)

            # calculate metrics
            batch_metrics = calculate_regression_metrics(outputs, masks)

            # update running statistics
            running_loss += v_loss.item()
            for key in running_metrics:
                running_metrics[key] += batch_metrics[key]

            # update progress bar
            pbar.set_postfix({
                'Loss': f'{v_loss.item():.4f}',
                'RMSE_Line': f'{batch_metrics["rmse_line"]:.4f}',
                'MAE_BG': f'{batch_metrics["mae_bg"]:.4f}'
            })

    # calculate epoch average validation loss and metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}

    return epoch_loss, epoch_metrics


def save_esl_validation_samples(model, dataloader, device, epoch, save_dir, num_samples=2):
    # save validation samples showing input, ground truth, and predictions
    model.eval()

    # create epoch-specific directory
    epoch_dir = Path(save_dir) / f'epoch_{epoch+1:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if saved_count >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # apply sigmoid activation to predictions
            outputs = torch.sigmoid(outputs)

            # convert to numpy for visualization
            for i in range(images.shape[0]):
                if saved_count >= num_samples:
                    break

                # get single sample
                input_img = images[i].cpu().squeeze().numpy()
                gt_mask = masks[i].cpu().squeeze().numpy()
                pred_mask = outputs[i].cpu().squeeze().numpy()

                # create figure with three subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # input depthmap
                axes[0].imshow(input_img, cmap='gray')
                axes[0].set_title('Input Depthmap')
                axes[0].axis('off')

                # ground truth mask
                axes[1].imshow(gt_mask, cmap='Reds', alpha=0.7)
                axes[1].imshow(input_img, cmap='gray', alpha=0.3)
                axes[1].set_title('Ground Truth SPL')
                axes[1].axis('off')

                # prediction mask
                axes[2].imshow(pred_mask, cmap='Blues', alpha=0.7)
                axes[2].imshow(input_img, cmap='gray', alpha=0.3)
                axes[2].set_title('Predicted SPL')
                axes[2].axis('off')

                plt.tight_layout()

                # save
                save_path = epoch_dir / f'sample_{saved_count+1:02d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                saved_count += 1

    print(f'Saved {saved_count} validation samples for epoch {epoch+1}')


def save_isl_validation_samples(model, dataloader, device, epoch, save_dir, num_samples=5):
    # Saves validation samples for the ISL regression task
    model.eval()
    epoch_dir = Path(save_dir) / f'epoch_{epoch+1:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            if saved_count >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)

            # Move to CPU and convert to numpy for visualization
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            outputs = outputs.cpu().numpy()

            for i in range(images.shape[0]):
                if saved_count >= num_samples:
                    break

                input_img = images[i, 0]
                gt_mask = masks[i, 0]
                pred_mask = outputs[i, 0]

                # Find a common color scale for ground truth and prediction
                vmin = min(gt_mask[gt_mask > 0].min(), pred_mask[pred_mask > 0].min()) if np.any(gt_mask > 0) and np.any(pred_mask > 0) else 0
                vmax = max(gt_mask.max(), pred_mask.max())

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Input Depthmap
                axes[0].imshow(input_img, cmap='gray')
                axes[0].set_title('Input Depthmap')
                axes[0].axis('off')

                # Ground Truth Line
                im1 = axes[1].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1].set_title('Ground Truth Line')
                axes[1].axis('off')
                axes[1].set_facecolor('black')


                # Predicted Line
                im2 = axes[2].imshow(np.ma.masked_where(pred_mask < 1e-3, pred_mask), cmap='viridis', vmin=vmin, vmax=vmax)
                axes[2].set_title('Predicted Line')
                axes[2].axis('off')
                axes[2].set_facecolor('black')

                plt.subplots_adjust(bottom=0.1, right=0.9, wspace=0.05)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                fig.colorbar(im2, cax=cbar_ax)
                
                save_path = epoch_dir / f'sample_{saved_count+1:02d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                saved_count += 1

    print(f'Saved {saved_count} validation samples for epoch {epoch+1} to {epoch_dir}')


def save_isl_validation_samples_only_the_prediction(model, dataloader, device, epoch, save_dir, num_samples=5):
    # Saves validation samples for the ISL regression task - predicted line only
    model.eval()
    epoch_dir = Path(save_dir) / f'epoch_{epoch+1:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            if saved_count >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)

            # Move to CPU and convert to numpy for visualization
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            outputs = outputs.cpu().numpy()

            for i in range(images.shape[0]):
                if saved_count >= num_samples:
                    break

                pred_mask = outputs[i, 0]

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))

                # Predicted Line only - with colors but no colorbar
                ax.imshow(np.ma.masked_where(pred_mask < 1e-3, pred_mask), cmap='viridis')
                ax.axis('off')
                ax.set_facecolor('black')
                
                save_path = epoch_dir / f'sample_{saved_count+1:02d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                saved_count += 1

    print(f'Saved {saved_count} validation samples for epoch {epoch+1} to {epoch_dir}')






    