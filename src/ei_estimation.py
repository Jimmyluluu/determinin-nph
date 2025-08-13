import os
import numpy as np
import SimpleITK as sitk
import math
import nibabel as nib

def read_rotation_matrix_from_flirt_mat(mat_path):
    """
    Reads the 3x3 rotation matrix from a FLIRT .mat file.
    """
    try:
        matrix = np.loadtxt(mat_path)
        return matrix[:3, :3]  # only the rotational part
    except Exception as e:
        print(f"Error reading matrix: {e}")
        return None

def rotation_matrix_to_euler_angles(R):
    """
    Converts a 3x3 rotation matrix to Euler angles (ZYX: yaw → pitch → roll), in degrees.
    """
    if abs(R[2, 0]) != 1:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        yaw = 0
        if R[2, 0] == -1:
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def extract_euler_angles_from_flirt_mat(mat_path):
    """
    Main function: reads a .mat file and returns Euler angles (roll, pitch, yaw) in degrees.
    """
    R = read_rotation_matrix_from_flirt_mat(mat_path)
    if R is None:
        return None
    return rotation_matrix_to_euler_angles(R)

def apply_euler_rotation(input_path, output_path, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, is_mask=True):
    """
    Rotates a 3D image or mask around its center by given Euler angles (roll, pitch, yaw) in degrees.
    
    Parameters:
        input_path (str): path to the input .nii.gz file
        output_path (str): path to save the rotated file
        roll_deg (float): rotation angle around X axis (roll) in degrees
        pitch_deg (float): rotation angle around Y axis (pitch) in degrees
        yaw_deg (float): rotation angle around Z axis (yaw) in degrees
        is_mask (bool): if True — use nearest neighbor interpolation and binarization (for masks), otherwise linear interpolation
    """
    # Load image
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Rotation center — physical center of the image
    center_index = [sz / 2.0 for sz in image.GetSize()]
    center_phys = image.TransformContinuousIndexToPhysicalPoint(center_index)

    # Create Euler 3D rotation transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_phys)

    # Convert degrees to radians
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)

    # Set rotation around three axes (X, Y, Z)
    transform.SetRotation(roll_rad, pitch_rad, yaw_rad)

    # Choose interpolation method
    if is_mask:
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    # Apply transformation with resampling
    resampled = sitk.Resample(
        image,
        image,      # output image size and space same as input
        transform,
        interp,
        0.0,        # background value for empty regions
        image.GetPixelID()
    )

    # If mask — binarize the result to remove interpolation artifacts
    if is_mask:
        resampled = sitk.BinaryThreshold(resampled, lowerThreshold=0.5, upperThreshold=10000, insideValue=1, outsideValue=0)

    # Save result
    sitk.WriteImage(resampled, output_path)
    print(f"Saved rotated image/mask to: {output_path}")

def find_best_ventricle_segment(nii_path: str, occupancy_threshold=0.9):
    """
    Find the segment with the maximum width and occupancy along the X-axis in a 3D ventricle mask.

    Parameters:
        nii_path (str): path to the NIfTI file containing the ventricle mask
        occupancy_threshold (float): minimum fraction of occupied pixels to consider the segment

    Returns:
        dict with parameters of the best segment: width, z, y, x1, x2, occupancy
    """
    img = nib.load(nii_path)
    mask_data = img.get_fdata()
    binary = (mask_data > 0).astype(np.uint8)

    best = {'width': 0, 'z': None, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}
    X, Y, Z = binary.shape

    for z in range(Z):
        slice_ = binary[:, :, z]
        for y in range(Y):
            col = slice_[:, y]
            xs = np.where(col > 0)[0]
            if xs.size < 2:
                continue
            x1, x2 = xs.min(), xs.max()
            width = x2 - x1
            if width <= best['width']:
                continue
            occupancy = col[x1:x2+1].sum() / (width + 1)
            if occupancy >= occupancy_threshold:
                best.update({'width': width, 'z': z, 'y': y, 'x1': int(x1), 'x2': int(x2), 'occupancy': occupancy})

    return best

def find_skull_segment(skull_path: str, z_fixed: int, y_fixed: int):
    """
    Find the width and coordinates of the skull segment at a given slice and column.

    Parameters:
        skull_path (str): path to the NIfTI file with the skull mask
        z_fixed (int): slice index along the Z-axis
        y_fixed (int): column index along the Y-axis

    Returns:
        dict with segment parameters: width, x1, x2, occupancy, z, y
    """
    skull_img = nib.load(skull_path)
    skull_data = skull_img.get_fdata()
    skull_binary = (skull_data > 0).astype(np.uint8)

    slice_skull = skull_binary[:, :, z_fixed]
    col_skull = slice_skull[:, y_fixed]

    xs = np.where(col_skull > 0)[0]
    if xs.size < 2:
        raise RuntimeError(f"No continuous skull segment found in column y={y_fixed}, slice z={z_fixed}")

    x1_skull, x2_skull = xs.min(), xs.max()
    width_skull = x2_skull - x1_skull
    occupancy_skull = col_skull[x1_skull:x2_skull+1].sum() / (width_skull + 1)

    return {
        'width': width_skull,
        'x1': int(x1_skull),
        'x2': int(x2_skull),
        'occupancy': occupancy_skull,
        'z': z_fixed,
        'y': y_fixed
    }

def check_hydrocephalus(ventricles_width, skull_width):
    """
    Calculate the Evans Index and provide a conclusion.

    Parameters:
        ventricles_width (float): width of the ventricles
        skull_width (float): width of the skull

    Returns:
        str with the result
    """
    if skull_width == 0:
        return "Error: denominator (skull width) cannot be zero."

    evans_index = ventricles_width / skull_width
    result = f"Evans Index: {evans_index:.3f}\n"

    if evans_index > 0.3:
        result += "Possible hydrocephalus (index > 0.3)."
    else:
        result += "Index is within normal range (≤ 0.3), hydrocephalus is unlikely."

    return result

def run_hydrocephalus_analysis(ventricle_nii_path, skull_nii_path, z_fixed=None, y_fixed=None, occupancy_threshold=0.9):
    """
    Main function to run the hydrocephalus analysis:
    - Find the best ventricle segment
    - Use its coordinates to analyze the skull
    - Calculate the Evans Index and output the result

    Parameters:
        ventricle_nii_path (str): path to the ventricle mask
        skull_nii_path (str): path to the skull mask
        z_fixed (int or None): fixed slice index for skull analysis (if None, taken from best ventricle segment)
        y_fixed (int or None): fixed column index for skull analysis (if None, taken from best ventricle segment)
        occupancy_threshold (float): minimum occupancy threshold for ventricles

    Returns:
        dict with analysis results:
            best_ventricle_segment,
            skull_segment,
            hydrocephalus_result
    """
    best_ventricle_segment = find_best_ventricle_segment(ventricle_nii_path, occupancy_threshold)

    # If z_fixed or y_fixed are not provided, use those from the best ventricle segment
    if z_fixed is None:
        z_fixed = best_ventricle_segment['z']
    if y_fixed is None:
        y_fixed = best_ventricle_segment['y']

    skull_segment = find_skull_segment(skull_nii_path, z_fixed, y_fixed)

    hydrocephalus_result = check_hydrocephalus(
        ventricles_width=best_ventricle_segment['width'],
        skull_width=skull_segment['width']
    )

    return {
        'best_ventricle_segment': best_ventricle_segment,
        'skull_segment': skull_segment,
        'hydrocephalus_result': hydrocephalus_result
    }

def check_and_get_paths(base_path: str):
    """
    Checks for the presence of three required files in base_path:
    - aligning.mat
    - brain_mask_combined.nii.gz
    - lateral_ventricles_combined.nii.gz

    If all three files are found, returns a dictionary with their paths
    and additionally adds expected paths for the aligned files.
    """
    expected_files = {
        "mat_file": "aligning.mat",
        "brain_mask_combined": "brain_mask_combined.nii.gz",
        "ventricles_combined": "lateral_ventricles_combined.nii.gz"
    }

    paths = {}
    missing_files = []

    for key, filename in expected_files.items():
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            paths[key] = full_path
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        success = False
    else:
        print("✅ All files found successfully!")
        success = True
        # Add expected paths for aligned files
        paths["aligned_brain"] = os.path.join(base_path, "brain_mask_aligned.nii.gz")
        paths["aligned_ventricles"] = os.path.join(base_path, "lateral_ventricles_aligned.nii.gz")

    return paths, success

def ei_estimation_pipeline(base: str, verbosity: bool = False):
    """
    Runs the full pipeline for hydrocephalus analysis from a base directory.

    Steps:
    - Check and get necessary file paths in the base folder
    - Extract Euler angles from FLIRT matrix file
    - Apply Euler rotations to brain mask and ventricles mask
    - Run hydrocephalus analysis on aligned images

    Args:
        base (str): Path to the base directory containing required files.
        verbosity (bool): If True, prints intermediate information. Default is False.

    Returns:
        dict: Results from the hydrocephalus analysis.
    """
    # Check for required files and get paths
    paths_dict, is_success = check_and_get_paths(base)
    if not is_success:
        raise FileNotFoundError("Required files not found in folder: " + base)
    
    if verbosity:
        print("Found paths:", paths_dict)
    
    # Extract Euler angles from the FLIRT matrix
    mat_file = paths_dict['mat_file']
    roll, pitch, yaw = extract_euler_angles_from_flirt_mat(mat_file)
    if verbosity:
        print(f"Roll: {roll:.2f}°\nPitch: {pitch:.2f}°\nYaw: {yaw:.2f}°")
    
    # Apply Euler rotation to brain mask
    apply_euler_rotation(
        input_path=paths_dict['brain_mask_combined'],
        output_path=paths_dict['aligned_brain'],
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw
    )
    # Apply Euler rotation to ventricles mask
    apply_euler_rotation(
        input_path=paths_dict['ventricles_combined'],
        output_path=paths_dict['aligned_ventricles'],
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw
    )
    
    # Run hydrocephalus analysis
    results = run_hydrocephalus_analysis(
        ventricle_nii_path=paths_dict['aligned_ventricles'],
        skull_nii_path=paths_dict['aligned_brain'],
        occupancy_threshold=0.9
    )
    
    if verbosity:
        print("Best ventricle segment:")
        print(results['best_ventricle_segment'])
        print("\nSkull segment:")
        print(results['skull_segment'])
        print("\nHydrocephalus check result:")
        print(results['hydrocephalus_result'])
    
    return results

if __name__ == "__main__":

    base = "/Users/maratorozaliev/Desktop/MindScope/data/_131505/"
    results = ei_estimation_pipeline(base)

    print("Best ventricle segment:")
    print(results['best_ventricle_segment'])
    print("\nSkull segment:")
    print(results['skull_segment'])
    print("\nHydrocephalus check result:")
    print(results['hydrocephalus_result'])