import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from convolutional_neural_network.core.cnn_data import DepthmapDataset
from convolutional_neural_network.core.model import UNet, R2AttUNet
from convolutional_neural_network.core.training import * 
from convolutional_neural_network.core.post_processing import *
from convolutional_neural_network.esl.config.base_config import BaseConfig
from convolutional_neural_network.esl.config.silver_config import SilverConfig
from convolutional_neural_network.esl.config.transfer_config import TransferConfig


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


def save_single_validation_sample(image, mask, output, original_filename, output_dir):
    # save a single validation sample showing input, ground truth, and prediction
    
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
    # convert to numpy for visualization (remove batch dimension)
    input_img = image[0].cpu().squeeze().numpy()  # (H, W)
    gt_mask = mask[0].cpu().squeeze().numpy()     # (H, W)
    pred_mask = output.cpu().squeeze().numpy() # (H, W)
    
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
    
    # create output filename 
    viz_filename = original_filename
    save_path = output_dir / viz_filename
    
    # save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {save_path}")


def main():
    # Initialize configuration
    base_config = BaseConfig()
    silver_config = SilverConfig()
    transfer_config = TransferConfig()

    # Set random seed
    set_random_seed(base_config.random_seed)

    # Create dataset
    base_dataset = DepthmapDataset(
        images_dir=base_config.depthmap_path,
        masks_dir=base_config.esl_mask_path,
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

    # validate the splits
    base_dataset.validate_split(
        train_indices=gold_train_indices,
        test_indices=gold_test_indices
    )
    base_dataset.validate_split(
        train_indices=silver_train_indices,
        test_indices=silver_test_indices
    )

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


    # Setup model, load pretrained weights
    pretrained_model_path = r"path/to/pretrained/model"
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {pretrained_model_path}")

        # Log pretrained model info
        pretrained_epoch = checkpoint.get('epoch', 'unknown')
        pretrained_val_dice = checkpoint.get('best_val_dice', 'unknown')

        print(f"Pretrained model loaded: Epoch {pretrained_epoch}, Best Val Dice {pretrained_val_dice:.4f}")
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}")


    model.eval()

    # average rmse
    total_rmse = 0.0
    total_samples = 0

    dataset_type_dict = {
        'full': full_test_dataset,
        'silver': silver_test_dataset,
        'gold': gold_test_dataset
    }
    dataset_type = 'full'  # 'silver', 'gold', 'full'
    dataset = dataset_type_dict[dataset_type]

    # Create output directory
    project_root = Path(__file__).resolve().parents[1]
    esl_validation_2d_mask_dir = project_root / 'esl_results' / 'silver' / f"esl_validation_2d_mask_{dataset_type}" / base_config.loss_function
    # or if the model is a fine-tuned one on the gold dataset:
    # esl_validation_combined_ply_dir = project_root / 'esl_results' / 'transfer' / f"{transfer_config.transfer_strategy}" / f"esl_validation_2d_mask_{dataset_type}" / base_config.loss_function
    esl_validation_2d_mask_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL validation 2d mask results to: {esl_validation_2d_mask_dir}")

    esl_validation_3d_coord_dir = project_root / 'esl_results' / 'silver' / f"esl_validation_3d_coordinates_{dataset_type}" / base_config.loss_function
    # or if the model is a fine-tuned one on the gold dataset:
    # esl_validation_combined_ply_dir = project_root / 'esl_results' / 'transfer' / f"{transfer_config.transfer_strategy}" / f"esl_validation_3d_coordinates_{dataset_type}" / base_config.loss_function
    esl_validation_3d_coord_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL validation 3D coordinates results to: {esl_validation_3d_coord_dir}")


    with torch.no_grad():
        for sample_idx in tqdm(range(len(dataset)), desc="Saving 3D coordinates for best model"):

            # get single sample
            image, mask = dataset[sample_idx]

            # add batch dimension and move to device
            image = image.unsqueeze(0).to(device)  # (1, C, H, W)
            mask = mask.unsqueeze(0).to(device)    # (1, C, H, W)

            # forward pass
            with torch.amp.autocast('cuda'):
                output = model(image)   
            
            # get original file name
            original_filename = dataset.get_original_filename(sample_idx)

            # clean up the predicted mask
            pred_binary = (torch.sigmoid(output) > 0.5).float().cpu()
            pred_binary = clean_esl_predicted_mask(pred_binary)

            # calculate metrics for this single sample
            sample_metrics = calculate_segmentation_metrics(output, mask)

            # save single validation sample visualization
            save_single_validation_sample(image, mask, pred_binary, original_filename, esl_validation_2d_mask_dir)

            # remove extension and create output file name
            json_filename = original_filename.replace('.png', '.json')

            # get backscan paths
            # get the original filename from dataset
            original_filename = dataset.get_original_filename(sample_idx)

            if original_filename is None:
                print(f"Could not get original filename for sample {sample_idx}")
                return None, None
            
            json_path, pointcloud_path = get_backscan_paths(original_filename, base_config.backscans_base_dir)

            if json_path and pointcloud_path:
                # load ground truth ESL
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        gt_data = json.load(f)
                except UnicodeDecodeError:
                    with open(json_path, 'r', encoding='latin-1') as f:
                        gt_data = json.load(f)

                # extract ground truth ESL coordinates
                if 'esl' in gt_data:
                    if 'formetric' in gt_data['esl']:
                        gt_3d = np.array(gt_data['esl']['formetric'])
                    else:
                        gt_3d = np.array(gt_data['esl']['pcdicomapp'])
                else:
                    gt_3d = np.array([])

                # convert prediction to 3D (without smoothing first - apply smoothing after determining target length)
                pred_3d = esl_mask_to_3d_from_tensors(
                    pred_binary, 
                    pointcloud_path, 
                    extract_line=True, 
                    method='skeleton'
                )


                if len(pred_3d) > 0 and len(gt_3d) > 0:
                    # determine target number of points more conservatively
                    target_num_points = determine_target_points(len(gt_3d), len(pred_3d))

                    print(f"Sample {original_filename}: GT {len(gt_3d)} -> {target_num_points}, Pred {len(pred_3d)} -> {target_num_points}")

                    # apply smoothing to both GT and prediction with same number of output points
                    gt_3d_smoothed = apply_spline_smoothing(
                        gt_3d, 
                        smoothing_factor=0.05,  
                        num_points=target_num_points
                    )
                    
                    pred_3d_smoothed = apply_spline_smoothing(
                        pred_3d, 
                        smoothing_factor=0.15,  
                        num_points=target_num_points
                    )

                    # calculate RMSE between smoothed curves
                    rmse = calculate_rmse(gt_3d_smoothed, pred_3d_smoothed, original_filename)

                    if rmse == float('inf'):
                        print(f"Sample {original_filename}: Invalid RMSE, skipping")
                        continue

                    # for calculating average RMSE
                    total_rmse += rmse
                    total_samples += 1
                    
                    # save prediction with original filename
                    pred_data = {
                        'dice_score': sample_metrics['dice'],
                        'iou_score': sample_metrics['iou'],
                        'pixel_accuracy': sample_metrics['pixel_accuracy'],
                        'esl_predicted': pred_3d_smoothed.tolist(),
                        'esl_ground_truth': gt_3d_smoothed.tolist(),
                        'rmse_3d': float(rmse),
                        'num_points': target_num_points,
                        'original_filename': original_filename,
                        'smoothing_applied': True,
                        'gt_smoothing_factor': 0.05,
                        'pred_smoothing_factor': 0.15
                    }

                    pred_file = Path(esl_validation_3d_coord_dir) / json_filename
                    with open(pred_file, 'w') as f:
                        json.dump(pred_data, f, indent=2)
                        
                    print(f"Sample {original_filename}: RMSE = {rmse:.2f}mm, Points = {target_num_points}")
                else:
                    print(f"Warning: No 3D prediction for sample {original_filename}")
            else:
                print(f"Warning: No backscan paths for sample {original_filename}")

    # calculate and print average RMSE
    if total_samples > 0:
        average_rmse = total_rmse / total_samples
        print(f"Average RMSE: {average_rmse:.2f}mm over {total_samples} samples")



if __name__ == "__main__":
    main()