import numpy as np
import json
import cv2
import torch
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
import open3d as o3d
from skimage.morphology import skeletonize
from pathlib import Path

from skimage.morphology import skeletonize
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def apply_spline_smoothing(points_3d, smoothing_factor=0.1, num_points=None):
    if len(points_3d) < 4:
        return points_3d

    try:
        # Remove duplicates first with tolerance
        unique_points, unique_indices = np.unique(
            np.round(points_3d, decimals=3),  # Round to avoid floating point issues
            axis=0, 
            return_index=True
        )

        if len(unique_points) < 4:
            print(f"Warning: Too few unique points ({len(unique_points)}), returning original")
            return points_3d

        # Sort by original order to maintain curve topology
        unique_indices.sort()
        unique_points = points_3d[unique_indices]

        print(f"Removed {len(points_3d) - len(unique_points)} duplicate points")

        # Apply spline to clean data
        tck, u = splprep(unique_points.T, s=smoothing_factor * len(unique_points))

        if num_points is None:
            num_points = len(points_3d)

        u_new = np.linspace(0, 1, num_points)
        smoothed_points = splev(u_new, tck)
        result = np.column_stack(smoothed_points)

        print(f"Spline smoothing applied: {len(unique_points)} -> {len(result)} points")
        return result

    except Exception as e:
        print(f"Warning: Spline smoothing failed ({e}), returning original points")
        return points_3d


def create_combined_ply(json_path, pointcloud_path, original_filename, output_ply_path, 
                       downsample_factor=10):
    # simple function to combine backscan, GT ESL, and predicted ESL into one .ply file

    modified_output_path = Path(output_ply_path) / f'{original_filename}.ply'

    # load backscan
    pcd_backscan = o3d.io.read_point_cloud(pointcloud_path)
    backscan_points = np.asarray(pcd_backscan.points)
    # downsample backscan for performance
    backscan_points = backscan_points[::downsample_factor]

    # load ground truth and predicted ESL from a single JSON file
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

    # create and save .ply file 
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.io.write_point_cloud(str(modified_output_path), combined_pcd)
    print(f"Saved combined visualization to: {modified_output_path}")
    return combined_pcd


def get_backscan_paths(original_filename, backscans_base_dir):
    # map dataset sample index to corresponding backscan files
    # returns: tuple: (json_path, pointcloud_path) or (None, None) if not found

    # remove file extension
    base_name = original_filename.replace('.png', '')

    # parse the filename to get hospital, patient_id, and additional info
    parts = base_name.split('_')

    if len(parts) < 2:
        print(f"Invalid filename format: {original_filename}")
        return None, None

    hospital = parts[0]
    patient_id = parts[1]

    # handle special cases for folder structure
    if hospital == 'balgrist' and len(parts) >= 3:
        # balgrist_1_flex or balgrist_1_relax
        condition = parts[2]
        folder_name = f"{hospital}_{patient_id}_{condition}"
        base_filename = f"{hospital}_{patient_id}_{condition}"
    elif hospital == 'geneva':
        # geneva_279-01 or geneva_279-02
        folder_name = f"{hospital}_{patient_id}"
        base_filename = f"{hospital}_{patient_id}"
    else:
        # croatian_679, italian_124, ukbb_7902
        folder_name = f"{hospital}_{patient_id}"
        base_filename = f"{hospital}_{patient_id}"

    # construct paths
    folder_path = Path(backscans_base_dir) / folder_name
    json_path = folder_path / f"{base_filename}_metadata_processed.json"
    pointcloud_path = folder_path / f"{base_filename}_processed.ply"

    # check if files exist
    if json_path.exists() and pointcloud_path.exists():
        return str(json_path), str(pointcloud_path)
    else:
        print(f"Backscan files not found for {folder_name}:")
        print(f"  JSON: {json_path} (exists: {json_path.exists()})")
        print(f"  PLY: {pointcloud_path} (exists: {pointcloud_path.exists()})")
        return None, None


def clean_esl_predicted_mask(mask_tensor, distance_threshold=60):
    # keep the largest component and any components within distance_threshold pixels of it

    mask_np = mask_tensor.squeeze().cpu().numpy()
    mask_binary = (mask_np > 0.5).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)

    if num_labels <= 1:
        return torch.from_numpy(mask_binary).float()

    # Find largest component (skip background at index 0)
    largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Get pixels for largest component
    largest_pixels = np.column_stack(np.where(labels == largest_idx))

    components_to_keep = [largest_idx]

    for i in range(1, num_labels):
        if i == largest_idx:
            continue

        # Get pixels for current component
        component_pixels = np.column_stack(np.where(labels == i))

        # Calculate minimum distance between components
        min_dist = cdist(largest_pixels, component_pixels).min()

        if min_dist <= distance_threshold:
            components_to_keep.append(i)

    # Create output mask
    cleaned_mask = np.isin(labels, components_to_keep).astype(np.uint8)

    return torch.from_numpy(cleaned_mask).float()


def clean_isl_predicted_mask(pred_tensor, value_threshold=0.1, min_component_size=50, distance_threshold=80):
    # clean ISL prediction by keeping components that are likely part of the spinal line
    
    # Convert to numpy and create binary mask
    pred_np = pred_tensor.squeeze().cpu().numpy()
    binary_mask = (pred_np > value_threshold).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:
        return pred_tensor
    
    # Filter components by size and vertical alignment (spinal line should be roughly vertical)
    valid_components = []
    
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_component_size:
            continue
            
        # Get component pixels and check if it's vertically oriented
        component_mask = (labels == i)
        y_coords, x_coords = np.where(component_mask)
        
        if len(y_coords) > 0:
            # Calculate aspect ratio and vertical span
            height = y_coords.max() - y_coords.min() + 1
            width = x_coords.max() - x_coords.min() + 1
            
            # Prefer components that are more vertical than horizontal
            # and have reasonable vertical span for spinal segments
            if height >= width * 0.5 and height > 20:  # At least somewhat vertical
                centroid_y = centroids[i][1]
                centroid_x = centroids[i][0]
                
                valid_components.append({
                    'label': i,
                    'area': area,
                    'centroid': (centroid_x, centroid_y),
                    'height': height,
                    'y_coords': y_coords,
                    'x_coords': x_coords
                })
    
    if not valid_components:
        return torch.zeros_like(pred_tensor)
    
    # Sort by area (largest first)
    valid_components.sort(key=lambda x: x['area'], reverse=True)
    
    # Start with the largest component
    components_to_keep = [valid_components[0]['label']]
    kept_centroids = [valid_components[0]['centroid']]
    
    # Add components that are close to already kept components
    for comp in valid_components[1:]:
        comp_centroid = comp['centroid']
        
        # Check distance to any kept component
        min_dist = float('inf')
        for kept_centroid in kept_centroids:
            dist = np.sqrt((comp_centroid[0] - kept_centroid[0])**2 + 
                          (comp_centroid[1] - kept_centroid[1])**2)
            min_dist = min(min_dist, dist)
        
        # Also check if components are vertically aligned (typical for spine)
        vertically_aligned = False
        for kept_centroid in kept_centroids:
            x_diff = abs(comp_centroid[0] - kept_centroid[0])
            if x_diff < 20:  # Within reasonable horizontal range for spine
                vertically_aligned = True
                break
        
        if min_dist <= distance_threshold and vertically_aligned:
            components_to_keep.append(comp['label'])
            kept_centroids.append(comp_centroid)
    
    print(f"ISL cleaning: Found {len(valid_components)} valid components, keeping {len(components_to_keep)}")
    
    # Create cleaned mask
    cleaned_binary = np.isin(labels, components_to_keep).astype(np.uint8)
    
    # Apply cleaned mask to original prediction values
    cleaned_pred = pred_np * cleaned_binary
    
    return torch.from_numpy(cleaned_pred).float()


def align_isl_coordinates_lateral_plane(pred_3d, gt_3d):
    # align predicted ISL coordinates to ground truth on the lateral (Y-Z plane)
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
    # align predicted ISL coordinates to ground truth on the X-Y plane
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


def esl_mask_to_3d_from_tensors(mask_tensor, pointcloud_path, extract_line=True, method='skeleton'):
    # Convert ESL mask tensor directly to 3D coordinates without saving to disk

    # step 1: convert tensor to numpy and ensure it's binary
    mask_np = mask_tensor.squeeze().numpy()
    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

    # step 2: get pixels to convert (either all pixels or clean line)
    if extract_line:
        if method == 'skeleton':
            skeleton = skeletonize(mask_binary > 0)
            pixels = np.where(skeleton)
        elif method == 'contour':
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_points = largest_contour.reshape(-1, 2) # [col, row]
                pixels = (contour_points[:, 1], contour_points[:, 0]) # convert to (row, col)
            else:
                return np.array([]).reshape(0, 3)
    else:
        pixels = np.where(mask_binary > 0)

    if len(pixels[0]) == 0:
        return np.array([]).reshape(0, 3)

    # step 3: convert pixel coordinates to X,Y world coordinates
    pixel_size = 1.2  # mm per pixel
    cx, cy = 257, 257  # image center

    i_coords = pixels[0]  # row coordinates
    j_coords = pixels[1]  # column coordinates

    X = (j_coords - cx) * pixel_size
    Y = -(i_coords - cy) * pixel_size

    # step 4: load original point cloud to find Z coordinates
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    original_points = np.asarray(pcd.points)

    # step 5: find nearest Z coordinates using KD-tree
    tree = cKDTree(original_points[:, :2])
    xy_points = np.column_stack([X, Y])
    distances, indices = tree.query(xy_points, k=1)
    Z = original_points[indices, 2]

    # step 6: combine into 3D coordinates
    points_3d = np.column_stack([X, Y, Z])

    return points_3d


def isl_mask_to_3d_from_tensors_with_smoothing(mask_tensor, extract_line=True, method='skeleton', 
                               smooth_lateral=True, rdp_epsilon=2.0, loess_frac=0.3):
    # Convert ISL mask tensor with depth values directly to 3D coordinates with optional smoothing

    def ramer_douglas_peucker(points, epsilon):
        # Simplify line using Ramer-Douglas-Peucker algorithm
        def perpendicular_distance(point, line_start, line_end):
            if np.array_equal(line_start, line_end):
                return np.linalg.norm(point - line_start)
            return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

        def rdp_recursive(points, start_idx, end_idx, epsilon):
            if end_idx <= start_idx + 1:
                return [start_idx, end_idx]

            max_distance = 0
            max_idx = start_idx

            for i in range(start_idx + 1, end_idx):
                distance = perpendicular_distance(points[i], points[start_idx], points[end_idx])
                if distance > max_distance:
                    max_distance = distance
                    max_idx = i

            if max_distance > epsilon:
                left_points = rdp_recursive(points, start_idx, max_idx, epsilon)
                right_points = rdp_recursive(points, max_idx, end_idx, epsilon)
                return left_points[:-1] + right_points
            else:
                return [start_idx, end_idx]

        if len(points) < 3:
            return points

        indices = rdp_recursive(points, 0, len(points) - 1, epsilon)
        return points[indices]

    def local_regression_smooth(y_coords, z_coords, frac=0.3):
        # Apply LOESS-like smoothing using local polynomial regression
        n = len(y_coords)
        smoothed_z = np.zeros_like(z_coords)

        for i in range(n):
            window_size = max(5, int(frac * n))
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n, i + window_size // 2)

            local_y = y_coords[start_idx:end_idx]
            local_z = z_coords[start_idx:end_idx]

            if len(local_y) < 3:
                smoothed_z[i] = z_coords[i]
                continue

            try:
                poly_features = PolynomialFeatures(degree=2)
                y_poly = poly_features.fit_transform(local_y.reshape(-1, 1))

                reg = LinearRegression()
                reg.fit(y_poly, local_z)

                current_y_poly = poly_features.transform(y_coords[i].reshape(-1, 1))
                smoothed_z[i] = reg.predict(current_y_poly)[0]
            except:
                smoothed_z[i] = z_coords[i]

        return smoothed_z

    # Step 1: Convert tensor to numpy
    mask_np = mask_tensor.squeeze().numpy()

    # Step 2: Get pixels to convert (either all pixels or clean line)
    if extract_line:
        if method == 'skeleton':
            binary_mask = (mask_np > 0).astype(np.uint8)
            skeleton = skeletonize(binary_mask > 0)
            pixels = np.where(skeleton)
        elif method == 'contour':
            binary_mask = (mask_np > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_points = largest_contour.reshape(-1, 2)
                pixels = (contour_points[:, 1], contour_points[:, 0])
            else:
                return np.array([]).reshape(0, 3)
    else:
        pixels = np.where(mask_np > 0)

    if len(pixels[0]) == 0:
        return np.array([]).reshape(0, 3)

    # Step 3: Convert pixel coordinates to X,Y world coordinates
    pixel_size = 1.2  # mm per pixel
    cx, cy = 257, 257  # image center

    i_coords = pixels[0]  # row coordinates
    j_coords = pixels[1]  # column coordinates

    X = (j_coords - cx) * pixel_size
    Y = (i_coords - cy) * pixel_size

    # Step 4: Extract Z coordinates directly from the depth mask
    Z = mask_np[pixels] * 255  # depth values in mm

    # Step 5: Combine into 3D coordinates
    points_3d = np.column_stack([X, Y, Z])

    # Step 6: Apply smoothing to lateral plane (Y-Z) if requested
    if smooth_lateral and len(points_3d) > 5:
        try:
            # Extract Y-Z coordinates for lateral plane smoothing
            y_coords = points_3d[:, 1]  # Y coordinates
            z_coords = points_3d[:, 2]  # Z coordinates
            x_coords = points_3d[:, 0]  # X coordinates (preserve)

            # Sort by Y coordinate for proper smoothing
            sort_indices = np.argsort(y_coords)
            y_sorted = y_coords[sort_indices]
            z_sorted = z_coords[sort_indices]
            x_sorted = x_coords[sort_indices]

            # Apply RDP simplification to Y-Z plane
            yz_points = np.column_stack([y_sorted, z_sorted])

            # Only apply RDP if we have enough points and epsilon > 0
            if len(yz_points) > 3 and rdp_epsilon > 0:
                try:
                    simplified_points = ramer_douglas_peucker(yz_points, rdp_epsilon)

                    # Create mapping for simplified points
                    simplified_indices = []
                    for simp_point in simplified_points:
                        # Find closest original point
                        distances = np.sum((yz_points - simp_point)**2, axis=1)
                        closest_idx = np.argmin(distances)
                        simplified_indices.append(closest_idx)

                    # Use simplified indices
                    y_for_smoothing = y_sorted[simplified_indices]
                    z_for_smoothing = z_sorted[simplified_indices]
                    x_for_smoothing = x_sorted[simplified_indices]
                except:
                    # If RDP fails, use original sorted data
                    y_for_smoothing = y_sorted
                    z_for_smoothing = z_sorted
                    x_for_smoothing = x_sorted
            else:
                y_for_smoothing = y_sorted
                z_for_smoothing = z_sorted
                x_for_smoothing = x_sorted

            # Apply local regression smoothing to Z coordinates
            if len(y_for_smoothing) > 3:
                z_smoothed = local_regression_smooth(y_for_smoothing, z_for_smoothing, frac=loess_frac)

                # Reconstruct smoothed 3D coordinates
                points_3d = np.column_stack([x_for_smoothing, y_for_smoothing, z_smoothed])

        except Exception as e:
            # If smoothing fails, return original coordinates
            print(f"Warning: Smoothing failed ({e}), returning original coordinates")
            pass

    return points_3d


def determine_target_points(gt_len, pred_len, max_upsample_ratio=3.0):
    # Determine target number of points

    # If prediction is much longer than GT, don't upsample GT too aggressively
    if pred_len > gt_len * max_upsample_ratio:
        target_points = min(pred_len, int(gt_len * max_upsample_ratio))
    else:
        # Use the longer of the two, but cap at reasonable maximum
        target_points = min(max(gt_len, pred_len), 300)  # cap at 300 points

    return target_points


def calculate_rmse(gt_3d, pred_3d, sample_name, max_reasonable_rmse=1000.0):
    # Calculate RMSE with detailed logging for debugging
    if len(gt_3d) == 0 or len(pred_3d) == 0:
        print(f"{sample_name}: Empty arrays - GT: {len(gt_3d)}, Pred: {len(pred_3d)}")
        return float('inf')

    if len(gt_3d) != len(pred_3d):
        print(f"{sample_name}: Point count mismatch - GT: {len(gt_3d)}, Pred: {len(pred_3d)}")
        return float('inf')

    # Check for invalid coordinates
    gt_finite = np.all(np.isfinite(gt_3d))
    pred_finite = np.all(np.isfinite(pred_3d))
    if not (gt_finite and pred_finite):
        print(f"{sample_name}: Invalid coordinates - GT finite: {gt_finite}, Pred finite: {pred_finite}")
        return float('inf')

    # Check coordinate ranges
    gt_range = np.ptp(gt_3d, axis=0)  # range for each dimension
    pred_range = np.ptp(pred_3d, axis=0)
    if np.any(gt_range > 1000) or np.any(pred_range > 1000):  # >1m range seems unreasonable
        print(f"{sample_name}: Large coordinate range - GT: {gt_range}, Pred: {pred_range}")
        return float('inf')

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.sum((gt_3d - pred_3d)**2, axis=1)))

    if not np.isfinite(rmse) or rmse > max_reasonable_rmse:
        print(f"{sample_name}: Invalid RMSE: {rmse:.2f}mm")
        return float('inf')

    return rmse