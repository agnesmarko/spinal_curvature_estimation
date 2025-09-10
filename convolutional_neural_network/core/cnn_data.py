# data handling: custom dataset class, data augmentation
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

import albumentations as A

# augmentation class
class Augmenter:

    def __init__(self):
        # the rotation limit is fixed, todo: think about a more flexible approach
        self.rotation_transform = A.Rotate(limit=3, p=1.0)

    def rotate(self, image, mask):

        # converting to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)

        # applying transformation
        augmented = self.rotation_transform(image=image_np, mask=mask_np)

        # converting back to PIL images
        image_aug = Image.fromarray(augmented['image'])
        mask_aug = Image.fromarray(augmented['mask'])

        return image_aug, mask_aug


# custom dataset class for storing data from different hospitals
class DepthmapDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, augment_hospitals: list = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment_hospitals = augment_hospitals if augment_hospitals else []
        self.original_filenames = []

        # the images and masks are stored in different directories, therefore mapping is necessary for dataset creation
        # the structure of the filenames is hospital_patientId_....png in both directories

        # mapping the images and masks based on the hospital_patientId
        self.image_mask_pairs = []
        self.hospital_indices = defaultdict(list)
        self._build_original_dataset()

        self.augmenter = Augmenter()

    
    def _build_original_dataset(self):
        # image and mask filepaths
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        image_files.sort()  # sorting to ensure consistent order
        mask_files = [f for f in os.listdir(self.masks_dir) if f.endswith('.png')]
        mask_files.sort()
        mask_dict = {}

        for mask_file in mask_files:
            # extract hospital_patientId from mask filename
            parts = mask_file.split('_')
            hospital_name = parts[0]
            patient_id = parts[1]
            key = f"{hospital_name}_{patient_id}"
            mask_dict[key] = mask_file


        # match images with masks
        # keep the original pairs as 2-tuples for the base dataset
        for image_file in image_files:
            # extract hospital_patientId from image filename
            parts = image_file.split('_')
            hospital_name = parts[0]
            patient_id = parts[1]
            key = f"{hospital_name}_{patient_id}"

            if key in mask_dict:
                self.image_mask_pairs.append((image_file, mask_dict[key]))
                self.hospital_indices[hospital_name].append(len(self.image_mask_pairs) - 1)

                # store the original filename 
                self.original_filenames.append(image_file)
            else:
                print(f"Warning: No mask found for {image_file}")


    def __len__(self):
        return len(self.image_mask_pairs)
    
    
    def __getitem__(self, idx):
        # about keeping I/O operations in the __getitem__ method:
        # Loading images in __getitem__() is the standard approach and often preferable because:
        # Memory efficiency: Only loads what's needed
        # Parallelization: DataLoader workers can load in parallel
        # Flexibility: Allows for dynamic augmentation

        image_file, mask_file, should_augment = self.image_mask_pairs[idx]
        
        # loading images
        image_path = os.path.join(self.images_dir, image_file)
        mask_path = os.path.join(self.masks_dir, mask_file)
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # applying augmentations in __getitem__(): does it gonna slow down the training?
        # Why Augmentation in __getitem__() is Usually Fine:
        # Parallelization: DataLoader workers can apply augmentation in parallel
        # Speed: Modern augmentation libraries are quite fast
        # Memory efficiency: No need to store pre-augmented images

        
        # applying augmentation if this is marked as an augmented sample
        if should_augment:
            image, mask = self.augmenter.rotate(image, mask)
        
        # converting to tensors
        image = torch.from_numpy(np.array(image)).float()
        mask = torch.from_numpy(np.array(mask)).float()

        # normalize images and masks
        if image.max() > 1.0:
            image = image / 255.0

        if mask.max() > 1.0:
            mask = mask / 255.0 # normalize if mask values are 0-255

        # channel dimension for grayscale images
        if image.dim() == 2: # if grayscale (H, W)
            image = image.unsqueeze(0)  # add channel dimension (1, H, W)
        
        if mask.dim() == 2:  # if grayscale (H, W)
            mask = mask.unsqueeze(0)  # add channel dimension (1, H, W)
        
        return image, mask
    

    def get_original_filename(self, index):
        # return the original filename for a given index
        if hasattr(self, 'original_filenames') and index < len(self.original_filenames):
            return self.original_filenames[index]
        return None


    def create_subset(self, indices, is_training=False):
        # create a subset of the dataset with specified indices and training flag
        subset_dataset = DepthmapDataset.__new__(DepthmapDataset)
        subset_dataset.images_dir = self.images_dir
        subset_dataset.masks_dir = self.masks_dir
        subset_dataset.augment_hospitals = self.augment_hospitals
        subset_dataset.is_train = is_training
        subset_dataset.augmenter = self.augmenter

        # Initialize original_filenames list for subset
        subset_dataset.original_filenames = []

        if is_training:
            # for training: create expanded dataset with both original and augmented versions
            expanded_pairs = []
            for idx in indices:
                image_file, mask_file = self.image_mask_pairs[idx]
                hospital_name = image_file.split('_')[0]

                # always add original version
                expanded_pairs.append((image_file, mask_file, False))
                subset_dataset.original_filenames.append(self.original_filenames[idx])

                # add augmented version for specified hospitals
                if hospital_name in self.augment_hospitals:
                    expanded_pairs.append((image_file, mask_file, True))
                    # For augmented versions, store the original filename too
                    subset_dataset.original_filenames.append(self.original_filenames[idx])

            subset_dataset.image_mask_pairs = expanded_pairs
        else:
            # for test/validation: only original versions
            subset_dataset.image_mask_pairs = [(self.image_mask_pairs[i][0], 
                                            self.image_mask_pairs[i][1], False) 
                                            for i in indices]

            # Map original filenames for subset indices
            subset_dataset.original_filenames = [self.original_filenames[i] for i in indices]

        # rebuild hospital indices for the new subset
        subset_dataset.hospital_indices = defaultdict(list)
        for new_idx, (image_file, _, _) in enumerate(subset_dataset.image_mask_pairs):
            hospital_name = image_file.split('_')[0]
            subset_dataset.hospital_indices[hospital_name].append(new_idx)

        return subset_dataset


    def hospital_aware_split(self, test_size=0.1, random_state=42, hospital_subset=None):
        # helper function for dataset train-test split
        train_indices = []
        test_indices = []

        # if hospital_subset is provided, filter the dataset to only include those hospitals
        hospital_to_use = self.hospital_indices.keys() if hospital_subset is None \
                                    else hospital_subset
        
        for hospital in hospital_to_use:
            hospital_indices = self.hospital_indices[hospital]

            if hospital in ['balgrist', 'geneva']:
                # for balgrist and geneva, grouping samples by patient ID to ensure 
                # multiple samples from the same patient stay in the same split
                patient_groups = defaultdict(list)

                for idx in hospital_indices:
                    image_file, _ = self.image_mask_pairs[idx]
                    parts = image_file.split('_')

                    if hospital == 'balgrist':
                        # for balgrist: patient ID is directly in parts[1]
                        patient_id = parts[1]
                    elif hospital == 'geneva':
                        # for geneva: patient ID might have suffix like 'patientId-01'
                        # extract the base patient ID by removing the suffix
                        patient_id_with_suffix = parts[1]
                        if '-' in patient_id_with_suffix:
                            patient_id = patient_id_with_suffix.split('-')[0]
                        else:
                            patient_id = patient_id_with_suffix

                    patient_groups[patient_id].append(idx)

                # splitting at patient level, not sample level
                patient_ids = list(patient_groups.keys())
                train_patients, test_patients = train_test_split(
                    patient_ids,
                    test_size=test_size,
                    random_state=random_state
                )

                # collecting all indices for train and test patients
                for patient_id in train_patients:
                    train_indices.extend(patient_groups[patient_id])

                for patient_id in test_patients:
                    test_indices.extend(patient_groups[patient_id])

            else:
                # for other hospitals, simply splitting the indices
                # splitting each hospital's data separately
                hospital_train, hospital_test = train_test_split(
                    hospital_indices,
                    test_size=test_size,
                    random_state=random_state
                )

                train_indices.extend(hospital_train)
                test_indices.extend(hospital_test)

        return train_indices, test_indices
    

    def validate_split(self, train_indices, test_indices):
        # Validate that same patients from balgrist and geneva don't appear in both train and test sets
        train_patients = {'balgrist': set(), 'geneva': set()}
        test_patients = {'balgrist': set(), 'geneva': set()}

        for idx in train_indices:
            image_file, _ = self.image_mask_pairs[idx]
            parts = image_file.split('_')
            hospital = parts[0]

            if hospital == 'balgrist':
                patient_id = parts[1]
                train_patients['balgrist'].add(patient_id)
            elif hospital == 'geneva':
                patient_id_with_suffix = parts[1]
                if '-' in patient_id_with_suffix:
                    patient_id = patient_id_with_suffix.split('-')[0]
                else:
                    patient_id = patient_id_with_suffix
                train_patients['geneva'].add(patient_id)

        for idx in test_indices:
            image_file, _ = self.image_mask_pairs[idx]
            parts = image_file.split('_')
            hospital = parts[0]

            if hospital == 'balgrist':
                patient_id = parts[1]
                test_patients['balgrist'].add(patient_id)
            elif hospital == 'geneva':
                patient_id_with_suffix = parts[1]
                if '-' in patient_id_with_suffix:
                    patient_id = patient_id_with_suffix.split('-')[0]
                else:
                    patient_id = patient_id_with_suffix
                test_patients['geneva'].add(patient_id)

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

            

