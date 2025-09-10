import os
from pathlib import Path
from point_transformer.core.training import *
from point_transformer.config.silver_config import SilverConfig
from point_transformer.config.transfer_config import TransferConfig

import numpy as np
import json
import open3d as o3d


def create_combined_ply(json_path, pointcloud_path, original_filename, output_ply_path, 
                       downsample_factor=10):

    modified_output_path = Path(output_ply_path) / f'{original_filename}.ply'

    # load backscan
    pcd_backscan = o3d.io.read_point_cloud(pointcloud_path)
    backscan_points = np.asarray(pcd_backscan.points)
    # downsample backscan for performance
    backscan_points = backscan_points[::downsample_factor]

    # load ground truth and predicted ISL from a single JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # if UTF-8 fails, try with latin-1 encoding
        with open(json_path, 'r', encoding='latin-1') as f:
            data = json.load(f)

    # extract ground truth and predicted ESL
    gt_isl = np.array(data.get('isl_ground_truth', []))
    predicted_isl = np.array(data.get('isl_predicted', []))

    # combine all points
    all_points = []
    all_colors = []

    # add backscan points (gray)
    all_points.append(backscan_points)
    all_colors.append(np.tile([0.7, 0.7, 0.7], (len(backscan_points), 1)))

    # add GT ISL points (green)
    if len(gt_isl) > 0:
        all_points.append(gt_isl)
        all_colors.append(np.tile([0, 1, 0], (len(gt_isl), 1)))

    # add predicted ISL points (red)
    if len(predicted_isl) > 0:
        all_points.append(predicted_isl)
        all_colors.append(np.tile([1, 0, 0], (len(predicted_isl), 1)))

    # combine into single arrays
    if not all_points:
        print("Warning: No points to save.")
        return None
        
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)

    # create and save .ply file using the modified path
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.io.write_point_cloud(str(modified_output_path), combined_pcd)
    print(f"Saved combined visualization to: {modified_output_path}")
    return combined_pcd


def main():
    # Load configuration
    silver_config = SilverConfig()
    transfer_config = TransferConfig()

    json_files = r'path\to\json\files'

    # Create output directory
    dataset_type = 'silver'
    project_root = Path(__file__).resolve().parents[1]
    isl_validation_combined_ply_dir = project_root / 'isl_results' / 'silver' / f"isl_validation_combined_ply_{dataset_type}" / silver_config.loss_function
    # or if the json files were generated using a fine-tuned model:
    # isl_validation_combined_ply_dir = project_root / 'isl_results' / 'transfer' / f"{transfer_config.transfer_strategy}" / f"isl_validation_combined_ply_{dataset_type}" / transfer_config.loss_function
    isl_validation_combined_ply_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ISL validation combined PLY results to: {isl_validation_combined_ply_dir}")

    # going through all the json files 
    for json_file in os.listdir(json_files):
        # check if the file is a JSON file
        if not json_file.endswith('.json'):
            continue
        # path of the file
        json_file_path = os.path.join(json_files, json_file)

        # get original filename, json path and ply path from the json file
        original_filename = None
        ply_path = None
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            original_filename = data.get('original_filename', None)
            ply_path = data.get('ply_path', None)

        if original_filename is None or ply_path is None:
            print(f"Skipping {json_file} due to missing metadata.")
            continue

        # combine ply files
        create_combined_ply(json_file_path, ply_path, original_filename, isl_validation_combined_ply_dir)


if __name__ == "__main__":
    main()