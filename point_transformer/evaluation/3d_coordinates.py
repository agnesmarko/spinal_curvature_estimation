import os
from pathlib import Path
from point_transformer.core.pt_data import BackscanDataset
from point_transformer.core.model.model import RegressionPointTransformer
from point_transformer.core.training import *
from point_transformer.config.silver_config import SilverConfig
from point_transformer.config.transfer_config import TransferConfig

import torch
import random
import numpy as np
import json
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def set_random_seed(seed=42):
    # set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse_3d(gt_coords, pred_coords):
    # Calculate RMSE between two sets of 3D coordinates
    if gt_coords.shape != pred_coords.shape:
        return float('inf')

    squared_diffs = np.sum((gt_coords - pred_coords) ** 2, axis=1)
    mse = np.mean(squared_diffs)
    return np.sqrt(mse)


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


def prepare_single_sample_with_collate(data_dict, target, device):
    # Actual collate function for a single sample

    # Create a fake batch with one sample
    fake_batch = [(data_dict, target)]

    # Use your existing collate function
    collated_data, collated_target = custom_collate_fn(fake_batch)

    # Move to device
    collated_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in collated_data.items()}
    collated_target = collated_target.to(device)

    return collated_data, collated_target


def align_isl_coordinates_lateral_plane(pred_3d, gt_3d):
    # Align predicted ISL coordinates to ground truth on the lateral (Y-Z plane)
    if len(pred_3d) == 0 or len(gt_3d) == 0:
        return pred_3d
    
    pred_3d = np.array(pred_3d)
    gt_3d = np.array(gt_3d)
    
    def alignment_cost(translation_yz):
        # Apply translation only to Y and Z coordinates
        translated_pred = pred_3d.copy()
        translated_pred[:, 1] += translation_yz[0]  # Y translation
        translated_pred[:, 2] += translation_yz[1]  # Z translation
        
        # Calculate minimum distances
        distances = cdist(translated_pred, gt_3d)
        min_distances = np.min(distances, axis=1)
        
        return np.mean(min_distances**2)
    
    # Optimize translation
    result = minimize(alignment_cost, [0.0, 0.0], method='BFGS')
    
    if result.success:
        # Apply optimal translation
        aligned_pred_3d = pred_3d.copy()
        aligned_pred_3d[:, 1] += result.x[0]  # Y translation
        aligned_pred_3d[:, 2] += result.x[1]  # Z translation
        return aligned_pred_3d
    else:
        return pred_3d


def align_isl_coordinates_coronal_plane(pred_3d, gt_3d):
    # Align predicted ISL coordinates to ground truth on the X-Y plane.
    if len(pred_3d) == 0 or len(gt_3d) == 0:
        return pred_3d
    
    pred_3d = np.array(pred_3d)
    gt_3d = np.array(gt_3d)
    
    def alignment_cost(translation_xy):
        # Apply translation only to X and Y coordinates
        translated_pred = pred_3d.copy()
        translated_pred[:, 0] += translation_xy[0]  # X translation
        translated_pred[:, 1] += translation_xy[1]  # Y translation
        
        # Calculate minimum distances
        distances = cdist(translated_pred, gt_3d)
        min_distances = np.min(distances, axis=1)
        
        return np.mean(min_distances**2)
    
    # Optimize translation
    result = minimize(alignment_cost, [0.0, 0.0], method='BFGS')
    
    if result.success:
        # Apply optimal translation
        aligned_pred_3d = pred_3d.copy()
        aligned_pred_3d[:, 0] += result.x[0]  # X translation
        aligned_pred_3d[:, 1] += result.x[1]  # Y translation
        return aligned_pred_3d
    else:
        return pred_3d


def main():
    # Load configuration
    silver_config = SilverConfig()
    transfer_config = TransferConfig()

    # Set random seed for reproducibility
    set_random_seed(silver_config.random_seed)

    # Create dataset
    base_dataset = BackscanDataset(
        base_dir=silver_config.base_path,
        num_points=silver_config.backscan_num_points,
        sampling_method=silver_config.sampling_method,
        grid_size=silver_config.grid_size
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
        base_dataset.hospital_aware_split(test_size=0.1, random_state=silver_config.random_seed,
                                        hospital_subset=silver_config.hospitals)

    silver_train_dataset = base_dataset.create_subset(silver_train_indices)
    silver_test_dataset = base_dataset.create_subset(silver_test_indices)

    # === GOLD DATASET ===
    gold_train_indices, gold_test_indices = \
        base_dataset.hospital_aware_split(test_size=0.1, random_state=transfer_config.random_seed,
                                        hospital_subset=transfer_config.hospitals)

    gold_train_dataset = base_dataset.create_subset(gold_train_indices)
    gold_test_dataset = base_dataset.create_subset(gold_test_indices)

    # validate the split
    base_dataset.validate_split(
        train_indices=silver_train_indices,
        test_indices=silver_test_indices
    )


    # initializing the model
    model_dict = {
        'pt_regression': RegressionPointTransformer(
            in_channels=silver_config.in_channels,
            num_points=silver_config.out_num_points,
            output_channels=silver_config.out_channels,
        )
    }

    model = model_dict[silver_config.model_type]


     # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Move model to device
    model.to(device)

    # Load pretrained model
    pretrained_model_path = r"path\to\pretrained\model"
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {pretrained_model_path}")

        # Log pretrained model info
        pretrained_epoch = checkpoint.get('epoch', 'unknown')
        pretrained_val_mse = checkpoint.get('best_val_mse', 'unknown')

        print(f"Pretrained model loaded: Epoch {pretrained_epoch}, Best Val MSE {pretrained_val_mse:.4f}")
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}")


    # average rmse
    total_rmse = 0.0
    total_samples = 0

    # Select dataset type
    dataset_type_dict = {
        'full': full_test_dataset,
        'silver': silver_test_dataset,
        'gold': gold_test_dataset
    }
    dataset_type = 'silver'  # or 'gold', 'full'
    dataset = dataset_type_dict[dataset_type]


    # Create output directory
    project_root = Path(__file__).resolve().parents[1]
    isl_validation_3d_coord_dir = project_root / 'isl_results' / 'silver' / f"isl_validation_3d_coordinates_{dataset_type}" / silver_config.loss_function
    # or if the model is a fine-tuned one on the gold dataset:
    # esl_validation_combined_ply_dir = project_root / 'isl_results' / 'transfer' / f"{transfer_config.transfer_strategy}" / f"isl_validation_3d_coordinates_{dataset_type}" / transfer_config.loss_function
    isl_validation_3d_coord_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL validation 3D coordinates results to: {isl_validation_3d_coord_dir}")


    # Set only dropout and batchnorm to eval, keep rest in training
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


    with torch.no_grad():
        for sample_idx in tqdm(range(len(dataset)), desc="Saving ISL validation samples"):
            # Get individual sample
            inputs, outputs = dataset[sample_idx]
            inputs, outputs = prepare_single_sample_with_collate(inputs, outputs, device)

            # Get original filename (folder name) from dataset
            original_filename = dataset.folder_names[sample_idx]
            json_filename = f"{original_filename}.json"

            # Add batch dimension and move to device
            if isinstance(inputs, dict):
                processed_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        # Move ALL tensors to device, but don't add batch dimension
                        processed_inputs[key] = value.to(device)
                    elif key == 'grid_size':
                        # grid_size is a float, keep as is
                        processed_inputs[key] = value
                    else:
                        processed_inputs[key] = value
                inputs = processed_inputs
            else:
                inputs = inputs.unsqueeze(0).to(device)

            outputs = outputs.unsqueeze(0).to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                predictions = model(inputs)

            # Calculate metrics for this sample
            sample_metrics = calculate_regression_metrics(predictions, outputs)

            # Convert predictions and outputs back to CPU and remove batch dimension
            predictions_np = predictions.squeeze(0).cpu().numpy()  # Shape: (48,)
            outputs_np = outputs.squeeze(0).cpu().numpy()  # Shape: (48,)

            # Reshape to 3D coordinates (16 points x 3 coordinates)
            pred_3d_normalized = predictions_np.reshape(16, 3)  # Shape: (16, 3)
            gt_3d_normalized = outputs_np.reshape(16, 3)  # Shape: (16, 3)

            # Get normalization parameters for this specific sample
            coord_min = dataset.coord_mins[sample_idx]
            coord_range = dataset.coord_ranges[sample_idx]

            pred_3d = pred_3d_normalized * coord_range + coord_min
            gt_3d = gt_3d_normalized * coord_range + coord_min

            # align
            pred_3d = align_isl_coordinates_lateral_plane(pred_3d, gt_3d)
            pred_3d = align_isl_coordinates_coronal_plane(pred_3d, gt_3d)

            rmse_3d = calculate_rmse_3d(gt_3d, pred_3d)

            if rmse_3d == float('inf'):
                print(f"Sample {original_filename}: Invalid RMSE, skipping")
                continue
            else:
                print(f"Sample {original_filename}: RMSE {rmse_3d:.4f}")

            # Update running totals
            total_rmse += rmse_3d
            total_samples += 1

            # Prepare prediction data for JSON saving
            pred_data = {
                'mse': float(sample_metrics['mse']),
                'mae': float(sample_metrics['mae']),
                'isl_predicted': pred_3d.tolist(),
                'isl_ground_truth': gt_3d.tolist(),
                'rmse_3d': float(rmse_3d),
                'original_filename': original_filename,
                'json_path': dataset.json_paths[sample_idx],
                'ply_path': dataset.ply_paths[sample_idx]
            }

            # Save prediction results to JSON file
            pred_file_path = isl_validation_3d_coord_dir / json_filename
            with open(pred_file_path, 'w', encoding='utf-8') as f:
                json.dump(pred_data, f, ensure_ascii=False, indent=4)

            print(f"Saved ISL validation 3D coordinates results to: {pred_file_path}")

    # Calculate and print average RMSE
    if total_samples > 0:
        avg_rmse = total_rmse / total_samples
        print(f"Average RMSE for ISL validation 3D coordinates: {avg_rmse:.4f}")
        print(f"Total samples processed: {total_samples}")
        print(f"Total RMSE: {total_rmse:.4f}")
    else:
        print("No valid samples found for RMSE calculation.")



if __name__ == "__main__":
    main()