import os
import torch
import numpy as np
import json
import open3d as o3d
from pathlib import Path
from torch.utils.data import Dataset


import os
import torch
import numpy as np
import json
import open3d as o3d
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


class BackscanDataset(Dataset):
    def __init__(self, base_dir: str, num_points: int = 1024, sampling_method: str = 'random', grid_size: float = 0.1, task_type: str = 'esl_pred'):
        super().__init__()

        # base directory with individual folders for every patient
        self.base_dir = base_dir
        self.grid_size = grid_size
        self.task_type = task_type

        # number of points to sample from each point cloud
        self.num_points = num_points #  1024, 2048, 4096 or 8192

        # sampling method: 'random' or 'fps'
        # poisson doesn't guarantee exact output count, that's why i use fps (farthest point sampling) instead
        self.sampling_method = sampling_method

        # Initialize lists and dictionaries
        self.ply_paths = []
        self.json_paths = []
        self.patient_ids = []
        self.folder_names = []
        self.hospital_indices = defaultdict(list)

        # Normalization
        self.coord_mins = []  # Store coord_min for each sample
        self.coord_ranges = []  # Store coord_range for each sample

        # ply file paths, json file paths, patient IDs
        # hospital_indices: for every hospital, the index of the its patients in the list
        self._get_file_paths()


    def __len__(self):
        return len(self.ply_paths)
    

    def _get_file_paths(self):
        # going through all the folders in the base directory
        for patient_folder in os.listdir(self.base_dir):
            patient_path = os.path.join(self.base_dir, patient_folder)

            # the name of the patient's folder has a structure like hospital_patientID
            # in the case of the balgrist hospital, the folder name is hospital_patientID_{flex/relax} - every patient has two folders
            # in the case of the geneva hospital, patient ID might have suffix like 'patientId-01'

            if os.path.isdir(patient_path):
                # getting the hospital name and patient ID
                parts = patient_folder.split('_')
                hospital_name = parts[0]
                patient_id_raw = parts[1]

                if hospital_name == 'geneva':
                    # extract base patient ID by removing suffix
                    patient_id = patient_id_raw.split('-')[0] if '-' in patient_id_raw else patient_id_raw
                else:
                    patient_id = patient_id_raw

                # getting the ply and json file paths
                ply_path = os.path.join(patient_path, f'{patient_folder}_processed.ply')
                json_path = os.path.join(patient_path, f'{patient_folder}_metadata_processed.json')

                # appending the file paths and patient IDs to the lists
                if os.path.exists(ply_path) and os.path.exists(json_path):
                    self.ply_paths.append(ply_path)
                    self.json_paths.append(json_path)
                    self.patient_ids.append(patient_id)
                    self.folder_names.append(patient_folder)
                    self.hospital_indices[hospital_name].append(len(self.ply_paths) - 1)

    @staticmethod
    def _farthest_point_sampling(point_cloud, num_points):
        # guarantees well-distributed points by maximizing minimum distances
        n_points = len(point_cloud)
        if num_points >= n_points:
            return np.arange(n_points)

        selected = [np.random.randint(n_points)]
        distances = np.full(n_points, np.inf)

        for _ in range(num_points - 1):
            # update distances to nearest selected point
            last_point = point_cloud[selected[-1]]
            new_distances = np.linalg.norm(point_cloud - last_point, axis=1)
            distances = np.minimum(distances, new_distances)

            # select point with maximum distance to any selected point
            selected.append(np.argmax(distances))

        return np.array(selected)


    def __getitem__(self, idx):
        # load the point cloud from the ply file
        ply_path = self.ply_paths[idx]
        point_cloud = np.asarray(o3d.io.read_point_cloud(ply_path).points, dtype=np.float32) # shape (N, 3)

        # the current number of points
        current_num_points = point_cloud.shape[0]

        # if the number of points is larger than the required number
        if current_num_points > self.num_points:
            if self.sampling_method == 'fps':
                # use farthest point sampling
                indices = self._farthest_point_sampling(point_cloud, self.num_points)
            elif self.sampling_method == 'random':
                # randomly sample without replacement
                indices = np.random.choice(current_num_points, self.num_points, replace=False)
        else:
            # if the number of points is smaller than the required number, pad by sampling with replacement
            extra = np.random.choice(current_num_points, self.num_points - current_num_points, replace=True)
            indices = np.concatenate((np.arange(current_num_points), extra))
        
        point_cloud = point_cloud[indices, :]

        point_cloud_tensor = torch.from_numpy(point_cloud)

        # load the 16 isl points from the json file and flatten them
        json_path = self.json_paths[idx]
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)    

            if self.task_type == 'esl_pred':
                if 'sampledLines16' in json_data:
                    if 'eslPcdicomapp' in json_data['sampledLines16']:
                        isl_points = \
                            np.array(json_data['sampledLines16']['eslPcdicomapp'], dtype=np.float32).flatten()
                    else:
                        isl_points = \
                            np.array(json_data['sampledLines16']['eslFormetric'], dtype=np.float32).flatten()
                        
            elif self.task_type == 'isl_pred':
                if 'sampledLines16' in json_data:
                    if 'islPcdicomapp' in json_data['sampledLines16']:
                        isl_points = \
                            np.array(json_data['sampledLines16']['islPcdicomapp'], dtype=np.float32).flatten()
                    else:
                        isl_points = \
                            np.array(json_data['sampledLines16']['islFormetric'], dtype=np.float32).flatten()
            
            else:
                print(f"Unknown task type: {self.task_type}")


        # Calculate normalization parameters from ALL points (point cloud + ISL)
        # This ensures both input and output use the same coordinate system
        all_points = np.vstack([point_cloud, isl_points.reshape(-1, 3)])
        coord_min = all_points.min(axis=0)
        coord_max = all_points.max(axis=0)
        coord_range = coord_max - coord_min
        coord_range = np.where(coord_range > 0, coord_range, 1.0)

        # Save normalization parameters for later use
        self.coord_mins.append(coord_min)
        self.coord_ranges.append(coord_range)

        # Normalize both point cloud and ISL points using the same parameters
        point_cloud_normalized = (point_cloud - coord_min) / coord_range
        isl_points_normalized = (isl_points.reshape(-1, 3) - coord_min) / coord_range
        
        # Convert to tensors
        point_cloud_tensor = torch.from_numpy(point_cloud_normalized)
        isl_points_tensor = torch.from_numpy(isl_points_normalized.flatten())

        # Create data dict
        data_dict = {
            'coord': point_cloud_tensor,
            'feat': point_cloud_tensor,
            'grid_size': self.grid_size
        }

        return data_dict, isl_points_tensor


    def hospital_aware_split(self, test_size=0.1, random_state=42, hospital_subset=None):
        # split dataset ensuring patients from balgrist and geneva hospitals 
        # don't appear in both train and test sets
        train_indices = []
        test_indices = []

        # if hospital_subset is provided, filter the dataset to only include those hospitals
        hospitals_to_use = self.hospital_indices.keys() if hospital_subset is None else hospital_subset
        
        for hospital in hospitals_to_use:
            hospital_indices = self.hospital_indices[hospital]
            
            if not hospital_indices:
                continue
                
            if hospital in ['balgrist', 'geneva']:
                # for balgrist and geneva, use GroupShuffleSplit to ensure 
                # multiple samples from the same patient stay in the same split
                
                # get patient IDs for this hospital's samples
                hospital_patient_ids = [self.patient_ids[idx] for idx in hospital_indices]
                
                # use GroupShuffleSplit for patient-level splitting
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                
                # split based on patient groups
                train_idx, test_idx = next(gss.split(
                    X=hospital_indices,  # the sample indices
                    groups=hospital_patient_ids  # patient IDs as groups
                ))
                
                # convert back to original dataset indices
                hospital_train = [hospital_indices[i] for i in train_idx]
                hospital_test = [hospital_indices[i] for i in test_idx]
                
                train_indices.extend(hospital_train)
                test_indices.extend(hospital_test)
                
            else:
                # for other hospitals, use simple random split since no patient grouping needed
                
                hospital_train, hospital_test = train_test_split(
                    hospital_indices,
                    test_size=test_size,
                    random_state=random_state
                )
                
                train_indices.extend(hospital_train)
                test_indices.extend(hospital_test)

        return train_indices, test_indices


    def validate_split(self, train_indices, test_indices):
        # validate that balgrist and geneva patients don't appear in both train and test sets
        train_patients = {'balgrist': set(), 'geneva': set()}
        test_patients = {'balgrist': set(), 'geneva': set()}

        # extract patient IDs from train indices
        for idx in train_indices:
            folder_name = os.path.basename(os.path.dirname(self.ply_paths[idx]))
            parts = folder_name.split('_')
            hospital = parts[0]
            
            if hospital in ['balgrist', 'geneva']:
                patient_id = self.patient_ids[idx]
                train_patients[hospital].add(patient_id)

        # extract patient IDs from test indices  
        for idx in test_indices:
            folder_name = os.path.basename(os.path.dirname(self.ply_paths[idx]))
            parts = folder_name.split('_')
            hospital = parts[0]
            
            if hospital in ['balgrist', 'geneva']:
                patient_id = self.patient_ids[idx]
                test_patients[hospital].add(patient_id)

        # check for overlaps
        validation_passed = True
        for hospital in ['balgrist', 'geneva']:
            if len(train_patients[hospital]) == 0 and len(test_patients[hospital]) == 0:
                continue
                
            overlap = train_patients[hospital] & test_patients[hospital]
            if overlap:
                print(f"Warning: {hospital} patients in both train and test: {overlap}")
                validation_passed = False
            else:
                print(f"No {hospital} patient overlap between train and test sets")

        return validation_passed
    

    def create_subset(self, indices):
        # create a subset of the dataset with specified indices
        subset_dataset = BackscanDataset.__new__(BackscanDataset)
        subset_dataset.base_dir = self.base_dir
        subset_dataset.num_points = self.num_points
        subset_dataset.sampling_method = self.sampling_method
        subset_dataset.grid_size = self.grid_size
        
        # create subset lists
        subset_dataset.ply_paths = [self.ply_paths[i] for i in indices]
        subset_dataset.json_paths = [self.json_paths[i] for i in indices]
        subset_dataset.patient_ids = [self.patient_ids[i] for i in indices]
        subset_dataset.folder_names = [self.folder_names[i] for i in indices]

        # Copy normalization parameters
        subset_dataset.coord_mins = []
        subset_dataset.coord_ranges = []

        # rebuild hospital indices for the subset
        subset_dataset.hospital_indices = defaultdict(list)
        for new_idx, original_idx in enumerate(indices):
            folder_name = os.path.basename(os.path.dirname(self.ply_paths[original_idx]))
            hospital_name = folder_name.split('_')[0]
            subset_dataset.hospital_indices[hospital_name].append(new_idx)
        
        return subset_dataset