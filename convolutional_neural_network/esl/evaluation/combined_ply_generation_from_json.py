import json
import os
from pathlib import Path

from convolutional_neural_network.core.post_processing import get_backscan_paths, create_combined_ply
from convolutional_neural_network.esl.config.base_config import BaseConfig
from convolutional_neural_network.esl.config.silver_config import SilverConfig
from convolutional_neural_network.esl.config.transfer_config import TransferConfig


def main():
    # Initialize configuration
    base_config = BaseConfig()
    silver_config = SilverConfig()
    transfer_config = TransferConfig()

    json_files = r'path/to/json/files'

    # Create output directory
    dataset_type = 'full' # 'silver', 'gold', 'full'
    project_root = Path(__file__).resolve().parents[1]
    esl_validation_combined_ply_dir = project_root / 'esl_results' / 'transfer' / f"{transfer_config.transfer_strategy}" / f"esl_validation_combined_ply_{dataset_type}" / base_config.loss_function
    # or if the json files were generated using a pre-trained model:
    # esl_validation_combined_ply_dir = project_root / 'esl_results' / 'silver' / f"esl_validation_combined_ply_{dataset_type}" / base_config.loss_function
    esl_validation_combined_ply_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ESL validation combined PLY results to: {esl_validation_combined_ply_dir}")


    # going through all the json files in the final_validation_3d_dir
    for json_file in os.listdir(json_files):
        # check if the file is a JSON file
        if not json_file.endswith('.json'):
            continue
        # path of the file
        json_file_path = os.path.join(json_files, json_file)

        # get original filename from the json file
        original_filename = None
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            original_filename = data.get('original_filename', None)
            
            if original_filename:
                original_filename = original_filename.replace('.png', '')

        if original_filename is None:
            print(f"Skipping {json_file} due to missing original filename.")
            continue

        original_json_path, pointcloud_path = get_backscan_paths(original_filename, base_config.backscans_base_dir)
        if pointcloud_path is None or original_json_path is None:
            print(f"Skipping {json_file} due to missing backscan files.")
            continue

        # combined ply file path
        combined_ply_path = esl_validation_combined_ply_dir
        create_combined_ply(
            json_path=json_file_path,
            pointcloud_path=pointcloud_path,
            original_filename=original_filename,
            output_ply_path=combined_ply_path,
            downsample_factor=30,  # adjust as needed
        )


if __name__ == "__main__":
    main()