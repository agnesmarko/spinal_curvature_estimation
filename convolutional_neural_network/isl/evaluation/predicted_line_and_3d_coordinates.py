import os
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import json

from wandb import config
from convolutional_neural_network.isl.config.base_config import BaseConfig
from convolutional_neural_network.isl.config.silver_config import SilverConfig
from convolutional_neural_network.isl.config.transfer_config import TransferConfig
from convolutional_neural_network.core.model import UNet, R2AttUNet
from convolutional_neural_network.core.cnn_data import DepthmapDataset
from convolutional_neural_network.core.post_processing import *
from convolutional_neural_network.core.training import *




def set_random_seed(seed=42):
    # set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_single_isl_validation_sample(image, mask, output, original_filename, output_dir):
    # save a single validation sample showing input, ground truth, and prediction
    
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
    # convert to numpy for visualization (remove batch dimension)
    input_img = image[0].cpu().squeeze().numpy()  # (H, W)
    gt_mask = mask[0].cpu().squeeze().numpy()     # (H, W)
    pred_mask = output.cpu().squeeze().numpy()    # (H, W)
    
    # Find a common color scale for ground truth and prediction
    vmin = min(gt_mask[gt_mask > 0].min(), pred_mask[pred_mask > 0].min()) if np.any(gt_mask > 0) and np.any(pred_mask > 0) else 0
    vmax = max(gt_mask.max(), pred_mask.max())

    # create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Input Depthmap
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Depthmap')
    axes[0].axis('off')

    # Ground Truth Line
    im1 = axes[1].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth ISL')
    axes[1].axis('off')
    axes[1].set_facecolor('black')

    # Predicted Line
    im2 = axes[2].imshow(np.ma.masked_where(pred_mask < 1e-3, pred_mask), cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title('Predicted ISL')
    axes[2].axis('off')
    axes[2].set_facecolor('black')

    plt.subplots_adjust(bottom=0.1, right=0.9, wspace=0.05)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im2, cax=cbar_ax)
    
    # create output filename
    viz_filename = original_filename
    save_path = output_dir / viz_filename
    
    # save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved ISL visualization: {save_path}")


def main():
    # Initialize configuration
    base_config = BaseConfig()
    silver_config = SilverConfig()
    transfer_config = TransferConfig()

    # Set random seed
    set_random_seed(base_config.random_seed)

    # Create base dataset
    base_dataset = DepthmapDataset(
        images_dir=base_config.depthmap_path,
        masks_dir=base_config.isl_mask_path,
        augment_hospitals=base_config.augment_hospitals,
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
        base_dataset.hospital_aware_split(test_size=0.1, random_state=silver_config.random_seed,
                                        hospital_subset=silver_config.hospitals)

    silver_train_dataset = base_dataset.create_subset(silver_train_indices, is_training=True)
    silver_test_dataset = base_dataset.create_subset(silver_test_indices, is_training=False)

    # == GOLD DATASET ==
    gold_train_indices, gold_test_indices = \
        base_dataset.hospital_aware_split(test_size=0.1, random_state=transfer_config.random_seed,
                                        hospital_subset=transfer_config.hospitals)

    gold_train_dataset = base_dataset.create_subset(gold_train_indices, is_training=True)
    gold_test_dataset = base_dataset.create_subset(gold_test_indices, is_training=False)

    # print the sizes of the datasets
    print(f"Full Train Dataset Size: {len(full_train_dataset)}")
    print(f"Full Test Dataset Size: {len(full_test_dataset)}")
    print(f"Silver Train Dataset Size: {len(silver_train_dataset)}")
    print(f"Silver Test Dataset Size: {len(silver_test_dataset)}")
    print(f"Gold Train Dataset Size: {len(gold_train_dataset)}")
    print(f"Gold Test Dataset Size: {len(gold_test_dataset)}")

    # Initialize model
    model_dict = {
        'unet': UNet(
            in_channels=base_config.in_channels,
            out_channels=base_config.out_channels,
            start_channels=base_config.start_channels,
            depth=base_config.depth,
            mode=base_config.mode,
        ),
        'r2attunet': R2AttUNet(
            in_channels=base_config.in_channels,
            out_channels=base_config.out_channels,
            start_channels=base_config.start_channels,
            depth=base_config.depth,
            t=1,
            mode=base_config.mode,
            use_attention=True,
            dropout=0.3
        )
    }

    model = model_dict[base_config.model_type]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)

    # Load pretrained model
    pretrained_model_path = r"path/to/pretrained/model"
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {pretrained_model_path}")

        # Log pretrained model info
        pretrained_epoch = checkpoint.get('epoch', 'unknown')
        pretrained_val_mse = checkpoint.get('best_val_rmse_line', 'unknown')

        print(f"Pretrained model loaded: Epoch {pretrained_epoch}, Best Val MSE {pretrained_val_mse:.4f}")
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}")

    model.eval()

    # average rmse
    total_rmse = 0.0
    total_samples = 0

    # Select dataset type
    dataset_type_dict = {
        'full': full_test_dataset,
        'silver': silver_test_dataset,
        'gold': gold_test_dataset
    }
    dataset_type = 'full'  # or 'gold', 'full'
    dataset = dataset_type_dict[dataset_type]

    # Create output directory
    project_root = Path(__file__).resolve().parents[1]
    isl_validation_2d_mask_dir = project_root / 'isl_results' / 'transfer' / f"{transfer_config.transfer_strategy}" / f"isl_validation_2d_mask_{dataset_type}_threshold" / base_config.loss_function
    # or if the model is a fine-tuned one on the gold dataset:
    # isl_validation_combined_ply_dir = project_root / 'isl_results' / 'silver' / f"isl_validation_2d_mask_{dataset_type}_threshold" / base_config.loss_function
    isl_validation_2d_mask_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL validation 2d mask results to: {isl_validation_2d_mask_dir}")

    isl_validation_3d_coord_dir = project_root / 'isl_results' / 'transfer' / f"{transfer_config.transfer_strategy}" / f"isl_validation_3d_coordinates_{dataset_type}" / base_config.loss_function
    # or if the model is a fine-tuned one on the gold dataset:
    # isl_validation_combined_ply_dir = project_root / 'isl_results' / 'silver' / f"isl_validation_3d_coordinates_{dataset_type}" / base_config.loss_function
    isl_validation_3d_coord_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL validation 3D coordinates results to: {isl_validation_3d_coord_dir}")

    with torch.no_grad():
        for sample_idx in tqdm(range(len(dataset)), desc="Saving ISL validation samples"):
            # get single sample
            image, mask = dataset[sample_idx]

            # add batch dimension and move to device
            image = image.unsqueeze(0).to(device)  # (1, C, H, W)
            mask = mask.unsqueeze(0).to(device)    # (1, C, H, W)

            # forward pass
            with torch.amp.autocast('cuda'):
                output = model(image)

            # calculate metrics
            sample_metrics = calculate_regression_metrics(output, mask)

            # Set all pixel values below a threshold to 0
            output_cpu = output[0].cpu()  # Move to CPU and remove batch dimension

            # Create mask for values below threshold
            below_threshold_mask = output_cpu < 0.1

            # Count how many values will be changed
            num_changed = torch.sum(below_threshold_mask).item()
            print(f"Setting {num_changed} pixel values below 0.1 to 0")

            # Set values below a threshold to 0
            output_cpu[below_threshold_mask] = 0

            # Move back to device
            output = output_cpu.unsqueeze(0).to(device)

            pred = output.float().cpu()

            pred = clean_isl_predicted_mask(pred, value_threshold=0.1, min_component_size=50, distance_threshold=80)

            # get original file name
            original_filename = dataset.get_original_filename(sample_idx)

            # save single validation sample visualization
            save_single_isl_validation_sample(image, mask, pred, original_filename, isl_validation_2d_mask_dir)

            # remove extension and create output file name
            json_filename = original_filename.replace('.png', '.json')

            # get backscan paths
            original_filename = dataset.get_original_filename(sample_idx)

            if original_filename is None:
                print(f"Could not get original filename for sample {sample_idx}")
                return None, None

            json_path, pointcloud_path = get_backscan_paths(original_filename, base_config.backscans_base_dir)

            if json_path and pointcloud_path:
                # load ground truth ISL
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        gt_data = json.load(f)
                except UnicodeDecodeError:
                    with open(json_path, 'r', encoding='latin-1') as f:
                        gt_data = json.load(f)

                # extract ground truth ISL coordinates
                if 'isl' in gt_data:
                    if 'pcdicomapp' in gt_data['isl']:
                        gt_3d = np.array(gt_data['isl']['pcdicomapp'])
                    else:
                        gt_3d = np.array(gt_data['isl']['formetric'])
                else:
                    gt_3d = np.array([])

                pred_3d = isl_mask_to_3d_from_tensors_with_smoothing(
                            pred, 
                            extract_line=True, 
                            method='skeleton',
                            smooth_lateral=True,
                            rdp_epsilon=1.0,
                            loess_frac=0.2
                        )
                
                
                if len(pred_3d) > 0 and len(gt_3d) > 0:
                    # flip Y coordinates to match the coordinate system
                    pred_3d[:, 1] = pred_3d[:, 1] * -1

                    # determine target number of points more conservatively
                    target_num_points = determine_target_points(len(gt_3d), len(pred_3d))

                    print(f"Sample {original_filename}: GT {len(gt_3d)} -> {target_num_points}, Pred {len(pred_3d)} -> {target_num_points}")

                    # apply smoothing to both GT and prediction with same number of output points
                    gt_3d = apply_spline_smoothing(
                        gt_3d, 
                        smoothing_factor=0.05,  
                        num_points=target_num_points
                    )
                    
                    pred_3d = apply_spline_smoothing(
                        pred_3d, 
                        smoothing_factor=0.15,  
                        num_points=target_num_points
                    )

                    # Align predicted coordinates to ground truth on lateral plane and on coronal plane
                    pred_3d = align_isl_coordinates_lateral_plane(pred_3d, gt_3d)
                    pred_3d = align_isl_coordinates_coronal_plane(pred_3d, gt_3d)

                    rmse = calculate_rmse(gt_3d, pred_3d, original_filename)

                    if rmse == float('inf'):
                        print(f"Sample {original_filename}: Invalid RMSE, skipping")
                        continue
                    else:
                        print(f"Sample {original_filename}: RMSE {rmse:.4f}")

                    # for calculating average RMSE
                    total_rmse += rmse
                    total_samples += 1

                    # save prediction with original filename
                    pred_data = {
                        'rmse_line': sample_metrics['rmse_line'],
                        'mae_line': sample_metrics['mae_line'],
                        'mae_bg': sample_metrics['mae_bg'],
                        'isl_predicted': pred_3d.tolist(),
                        'isl_ground_truth': gt_3d.tolist(),
                        'rmse_3d': float(rmse),
                        'original_filename': original_filename,
                        'smoothing_applied': True,
                        'gt_smoothing_factor': 0.05,
                        'pred_smoothing_factor': 0.15
                    }

                    pred_file_path = Path(isl_validation_3d_coord_dir) / json_filename
                    with open(pred_file_path, 'w', encoding='utf-8') as f:
                        json.dump(pred_data, f, ensure_ascii=False, indent=4)

                    print(f"Saved ISL validation 3D coordinates results to: {pred_file_path}")

                else:
                    print(f"Sample {original_filename}: No valid 3D coordinates found")
            else:
                print(f"Sample {original_filename}: No backscan paths found, skipping 3D coordinates saving")
    
    # calculate and print average RMSE
    if total_samples > 0:
        average_rmse = total_rmse / total_samples
        print(f"Average RMSE for {dataset_type} dataset: {average_rmse:.4f}")

if __name__ == "__main__":
    main()
