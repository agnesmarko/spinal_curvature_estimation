import torch
import torch.nn as nn
import torch.optim as optim
import os

def setup_transfer_learning(model, config, device='cuda', esl_transfer=True):
    # Setup transfer learning from pretrained model

    # Load pretrained weights
    if os.path.exists(config.pretrained_model_path):
        checkpoint = torch.load(config.pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {config.pretrained_model_path}")

        # Log pretrained model info
        pretrained_epoch = checkpoint.get('epoch', 'unknown')
        if esl_transfer:
            pretrained_val_dice = checkpoint.get('best_val_dice', 'unknown')
            print(f"Pretrained model: epoch {pretrained_epoch}, val_dice {pretrained_val_dice}")
        else:
            pretrained_val_rmse = checkpoint.get('best_val_rmse_line', 'unknown')
            print(f"Pretrained model: epoch {pretrained_epoch}, val_rmse_line {pretrained_val_rmse}")
    else:
        print(f"Warning: Pretrained model not found at {config.pretrained_model_path}")
        print("Training from scratch")
        return model, None

    # Apply transfer learning strategy
    if config.transfer_strategy == 'freeze_encoder':
        model.freeze_encoder()

    elif config.transfer_strategy == 'freeze_decoder':
        model.freeze_decoder()

    elif config.transfer_strategy == 'freeze_early_layers':
        model.freeze_early_layers(num_layers=2)

    elif config.transfer_strategy == 'train_all':
        pass  # All layers trainable

    elif config.transfer_strategy == 'progressive':
        model.freeze_encoder()

    else:
        raise ValueError(f"Unknown strategy: {config.transfer_strategy}")

    # Optionally freeze BatchNorm layers
    if config.freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        print("BatchNorm layers frozen")

    model.print_layer_status()
    return model, checkpoint


def create_transfer_optimizer(model, config):
    # Create optimizer for transfer learning

    if config.transfer_strategy == 'train_all':
        # Different learning rates for encoder and decoder
        encoder_params = []
        decoder_params = []

        for block in model.down_blocks:
            encoder_params.extend([p for p in block.parameters() if p.requires_grad])

        for block in model.up_blocks:
            decoder_params.extend([p for p in block.parameters() if p.requires_grad])

        final_params = [p for p in model.final_conv.parameters() if p.requires_grad]

        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.initial_lr * config.encoder_lr_multiplier},
            {'params': decoder_params, 'lr': config.initial_lr * config.decoder_lr_multiplier},
            {'params': final_params, 'lr': config.initial_lr}
        ], weight_decay=config.weight_decay)

    else:
        # Standard optimizer for frozen strategies
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.initial_lr,
            weight_decay=config.weight_decay
        )

    return optimizer


def handle_progressive_unfreezing(model, config, epoch, optimizer):
    # Handle progressive unfreezing during training

    if (config.transfer_strategy == 'progressive' and 
        epoch == config.unfreeze_epoch):

        print(f"\n=== Progressive Unfreezing at Epoch {epoch} ===")
        model.unfreeze_encoder()

        # Create new optimizer with all parameters
        new_optimizer = optim.AdamW(
            model.parameters(),
            lr=config.initial_lr * config.unfreeze_lr_multiplier,
            weight_decay=config.weight_decay
        )

        print("Encoder unfrozen, optimizer recreated")
        model.print_layer_status()
        return new_optimizer

    return optimizer