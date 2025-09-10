import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
import torch.nn.functional as F

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
        

def calculate_regression_metrics(pred, target):
    # Calculate basic regression metrics for the ISL task

    # Ensure tensors have the same shape
    if target.dim() > pred.dim():
        # Squeeze extra dimensions from target
        target = target.squeeze()
    elif pred.dim() > target.dim():
        # Squeeze extra dimensions from pred
        pred = pred.squeeze()

    # Ensure both tensors have the same shape
    if pred.shape != target.shape:
        # If still different, try to reshape to match
        if target.numel() == pred.numel():
            target = target.reshape(pred.shape)


    # Overall MSE and MAE across all coordinates
    mae = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)

    
    return {
        'mse': mse.item(),
        'mae': mae.item()
    }


def train_epoch_isl(model, dataloader, criterion, optimizer, device, epoch, log_every_n_batches=5, scaler=None):
    model.eval()  # Temporarily set to eval mode to disable any stateful operations
    with torch.no_grad():
        # Clear any cached values
        torch.cuda.empty_cache()
        
    # training for one epoch for the ISL regression task
    model.train()

    # initialize scaler for mixed precision
    if scaler is None:
        scaler = torch.amp.GradScaler()

    # metrics for the entire epoch
    running_loss = 0.0
    running_metrics = {'mse': 0.0, 'mae': 0.0}

    # metrics for batch-level logging
    batch_running_loss = 0.0
    batch_running_metrics = {key: 0.0 for key in running_metrics}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} training')

    for batch_idx, (inputs, outputs) in enumerate(pbar):
        if isinstance(inputs, dict):
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value 
                for key, value in inputs.items()
            }
        else:
            inputs = inputs.to(device)
        
        outputs = outputs.to(device)

        optimizer.zero_grad()

        # forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            predictions = model(inputs)
            loss = criterion(predictions, outputs)

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
            batch_metrics = calculate_regression_metrics(predictions, outputs)

        # detailed logging for debugging
        if batch_idx % 10 == 0:
            print(f"\nBatch {batch_idx} Details:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  MSE: {batch_metrics['mse']:.6f}, MAE: {batch_metrics['mae']:.6f}")


        # update running metrics
        running_loss += loss.item()
        running_metrics['mse'] += batch_metrics['mse']
        running_metrics['mae'] += batch_metrics['mae']

        # update batch-level metrics
        batch_running_loss += loss.item()
        batch_running_metrics['mse'] += batch_metrics['mse']
        batch_running_metrics['mae'] += batch_metrics['mae']

        # update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'MSE': f'{batch_metrics["mse"]:.4f}',
            'MAE': f'{batch_metrics["mae"]:.4f}',
        })

        # log to wandb every N batches
        if (batch_idx + 1) % log_every_n_batches == 0:
            # calculate averages over the last N batches
            avg_batch_loss = batch_running_loss / log_every_n_batches
            avg_batch_metrics = {k: v / log_every_n_batches for k, v in batch_running_metrics.items()}

            # calculate global step for wandb
            global_step = epoch * len(dataloader) + batch_idx + 1

            # log to wandb
            wandb.log({
                'step': global_step,
                'batch_train_loss': avg_batch_loss,
                'batch_train_mse': avg_batch_metrics['mse'],
                'batch_train_mae': avg_batch_metrics['mae'],
                'batch_learning_rate': optimizer.param_groups[0]['lr'],
            })

            # reset batch-level metrics
            batch_running_loss = 0.0
            batch_running_metrics = {key: 0.0 for key in running_metrics}

    # calculate average loss and metrics for the epoch
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {k: v / len(dataloader) for k, v in running_metrics.items()}

    return epoch_loss, epoch_metrics


def validate_epoch_isl(model, dataloader, criterion, device, epoch):
    # empty cache
    torch.cuda.empty_cache()

    # Set only dropout and batchnorm to eval, keep rest in training
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


    running_loss = 0.0
    running_metrics = {'mse': 0.0, 'mae': 0.0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} validation')

    with torch.no_grad():
        for batch_idx, (inputs, outputs) in enumerate(pbar):
            if isinstance(inputs, dict):
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value 
                    for key, value in inputs.items()
                }
            else:
                inputs = inputs.to(device)
            
            outputs = outputs.to(device)

            # forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                predictions = model(inputs)
                loss = criterion(predictions, outputs)

            # calculate metrics
            batch_metrics = calculate_regression_metrics(predictions, outputs)

            # update running metrics
            running_loss += loss.item()
            running_metrics['mse'] += batch_metrics['mse']
            running_metrics['mae'] += batch_metrics['mae']

            # update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MSE': f'{batch_metrics["mse"]:.4f}',
                'MAE': f'{batch_metrics["mae"]:.4f}',
            })

    # calculate average loss and metrics for the epoch
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {k: v / len(dataloader) for k, v in running_metrics.items()}

    return epoch_loss, epoch_metrics
