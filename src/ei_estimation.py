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

def find_best_ventricle_segment(nii_path: str, occupancy_threshold=0.9, frontal_horn_only=True, frontal_ratio=0.4):
    """
    Find the segment with the maximum width and occupancy along the X-axis in a 3D ventricle mask.

    Medical Standard: Evans Index should be measured at the frontal horns (anterior horns)
    of the lateral ventricles, which are typically located in the anterior 1/3 of the brain.

    Parameters:
        nii_path (str): path to the NIfTI file containing the ventricle mask
        occupancy_threshold (float): minimum fraction of occupied pixels to consider the segment
        frontal_horn_only (bool): if True, only search in the anterior region (frontal horns)
        frontal_ratio (float): ratio of anterior region to search (0.4 means anterior 40%)

    Returns:
        dict with parameters of the best segment: width, z, y, x1, x2, occupancy,
              anatomical_region, total_z_range
    """
    img = nib.load(nii_path)
    mask_data = img.get_fdata()
    binary = (mask_data > 0).astype(np.uint8)

    best = {'width': 0, 'z': None, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}
    X, Y, Z = binary.shape

    # Define search range for frontal horns
    if frontal_horn_only:
        # Find the range where ventricles actually exist
        ventricle_slices = []
        for z in range(Z):
            if np.any(binary[:, :, z] > 0):
                ventricle_slices.append(z)

        if not ventricle_slices:
            raise ValueError("No ventricle voxels found in the image")

        ventricle_start = min(ventricle_slices)
        ventricle_end = max(ventricle_slices)
        ventricle_range = ventricle_end - ventricle_start

        # Search only in the anterior portion (frontal horns)
        search_start = ventricle_start
        search_end = ventricle_start + int(ventricle_range * frontal_ratio)
        search_range = range(search_start, min(search_end + 1, Z))

        anatomical_info = {
            'search_region': 'frontal_horns',
            'ventricle_z_range': (ventricle_start, ventricle_end),
            'search_z_range': (search_start, search_end),
            'frontal_ratio_used': frontal_ratio
        }
    else:
        search_range = range(Z)
        anatomical_info = {
            'search_region': 'full_ventricle',
            'ventricle_z_range': (0, Z-1),
            'search_z_range': (0, Z-1),
            'frontal_ratio_used': None
        }

    for z in search_range:
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

    # Add anatomical information to the result
    best.update(anatomical_info)

    if best['z'] is None:
        raise ValueError(f"No suitable ventricle segment found with occupancy >= {occupancy_threshold}")

    return best

def find_skull_segment(skull_path: str, z_fixed: int):
    """
    Find the widest skull segment across all y-columns at a given Z slice.

    Parameters:
        skull_path (str): path to the NIfTI file with the skull mask
        z_fixed (int): slice index along the Z-axis

    Returns:
        dict with segment parameters: width, x1, x2, occupancy, z, y
    """
    skull_img = nib.load(skull_path)
    skull_data = skull_img.get_fdata()
    skull_binary = (skull_data > 0).astype(np.uint8)

    slice_skull = skull_binary[:, :, z_fixed]

    best_segment = None

    for y in range(slice_skull.shape[1]):
        col_skull = slice_skull[:, y]
        xs = np.where(col_skull > 0)[0]

        if xs.size < 2:
            continue  # нет сегмента для этой колонки

        x1_skull, x2_skull = xs.min(), xs.max()
        width_skull = x2_skull - x1_skull
        occupancy_skull = col_skull[x1_skull:x2_skull+1].sum() / (width_skull + 1)

        if (best_segment is None) or (width_skull > best_segment['width']):
            best_segment = {
                'width': width_skull,
                'x1': int(x1_skull),
                'x2': int(x2_skull),
                'occupancy': occupancy_skull,
                'z': z_fixed,
                'y': y
            }

    if best_segment is None:
        raise RuntimeError(f"No skull segment found in slice z={z_fixed}")

    return best_segment

def validate_frontal_horn_anatomy(ventricle_segment_info, skull_segment_info):
    """
    Validate if the selected ventricle segment represents a proper frontal horn measurement
    according to medical standards for Evans Index.

    Parameters:
        ventricle_segment_info (dict): result from find_best_ventricle_segment
        skull_segment_info (dict): result from find_skull_segment

    Returns:
        dict: validation results with warnings and recommendations
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'recommendations': [],
        'confidence_score': 1.0
    }

    # Check if search was restricted to frontal horns
    if ventricle_segment_info.get('search_region') != 'frontal_horns':
        validation['warnings'].append("Search was not restricted to frontal horn region")
        validation['confidence_score'] -= 0.2

    # Check ventricle occupancy
    occupancy = ventricle_segment_info.get('occupancy', 0)
    if occupancy < 0.85:
        validation['warnings'].append(f"Low ventricle occupancy ({occupancy:.2f} < 0.85) - may indicate fragmented ventricle")
        validation['confidence_score'] -= 0.15

    # Check if measurement is in anterior portion
    if 'search_z_range' in ventricle_segment_info and 'ventricle_z_range' in ventricle_segment_info:
        search_start, search_end = ventricle_segment_info['search_z_range']
        ventricle_start, ventricle_end = ventricle_segment_info['ventricle_z_range']
        measured_z = ventricle_segment_info['z']

        # Check if measured slice is actually in anterior region
        relative_position = (measured_z - ventricle_start) / (ventricle_end - ventricle_start) if (ventricle_end > ventricle_start) else 0

        if relative_position > 0.5:
            validation['warnings'].append(f"Measurement at slice {measured_z} is in posterior region (relative position: {relative_position:.2f})")
            validation['confidence_score'] -= 0.3
        elif relative_position > 0.4:
            validation['warnings'].append(f"Measurement at slice {measured_z} is at the border of frontal horn region")
            validation['confidence_score'] -= 0.1

    # Check ventricle to skull width ratio (Evans Index range check)
    ventricle_width = ventricle_segment_info.get('width', 0)
    skull_width = skull_segment_info.get('width', 1)
    evans_index = ventricle_width / skull_width if skull_width > 0 else 0

    if evans_index > 0.6:
        validation['warnings'].append(f"Extremely high Evans Index ({evans_index:.3f}) - verify measurement accuracy")
        validation['confidence_score'] -= 0.2
    elif evans_index < 0.1:
        validation['warnings'].append(f"Extremely low Evans Index ({evans_index:.3f}) - may indicate measurement error")
        validation['confidence_score'] -= 0.2

    # Check for symmetry (approximate)
    ventricle_center = (ventricle_segment_info.get('x1', 0) + ventricle_segment_info.get('x2', 0)) / 2
    skull_center = (skull_segment_info.get('x1', 0) + skull_segment_info.get('x2', 0)) / 2
    center_offset = abs(ventricle_center - skull_center)

    if center_offset > skull_width * 0.1:  # More than 10% offset
        validation['warnings'].append(f"Ventricle center significantly offset from skull center (offset: {center_offset:.1f} pixels)")
        validation['confidence_score'] -= 0.1

    # Generate recommendations
    if validation['confidence_score'] < 0.7:
        validation['recommendations'].append("Consider manual verification of the measurement location")
        validation['is_valid'] = False

    if ventricle_segment_info.get('search_region') != 'frontal_horns':
        validation['recommendations'].append("Use frontal_horn_only=True for standard Evans Index measurement")

    if len(validation['warnings']) == 0:
        validation['recommendations'].append("Measurement appears to follow medical standards for Evans Index")

    return validation

def visualize_evans_index_measurement(
    ventricle_path, skull_path,
    ventricle_segment_info, skull_segment_info,
    validation_info=None,
    save_path=None
):
    """
    Enhanced visualization showing Evans Index measurement with anatomical validation.

    Parameters:
        ventricle_path (str): path to ventricle mask (.nii or .nii.gz)
        skull_path (str): path to skull mask
        ventricle_segment_info (dict): result from find_best_ventricle_segment
        skull_segment_info (dict): result from find_skull_segment
        validation_info (dict, optional): result from validate_frontal_horn_anatomy
        save_path (str, optional): path to save the visualization
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    slice_index = ventricle_segment_info['z']

    # Load and binarize both masks
    ventricle_mask = (nib.load(ventricle_path).get_fdata() > 0).astype(np.uint8)
    skull_mask = (nib.load(skull_path).get_fdata() > 0).astype(np.uint8)

    # Extract slices
    ventricle_slice = ventricle_mask[:, :, slice_index]
    skull_slice = skull_mask[:, :, slice_index]

    x_dim, y_dim = ventricle_slice.shape

    # Prepare meshgrid
    xs, ys = np.meshgrid(range(x_dim), range(y_dim), indexing='ij')
    xs = xs.flatten()
    ys = ys.flatten()

    # Ventricle points
    v_mask = ventricle_slice.flatten() > 0
    x_ventricles = xs[v_mask]
    y_ventricles = ys[v_mask]

    # Skull points
    s_mask = skull_slice.flatten() > 0
    x_skull = xs[s_mask]
    y_skull = ys[s_mask]

    # Create subplot
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Ventricles (Frontal Horns)', 'Brain Mask'),
                       horizontal_spacing=0.1)

    # Add ventricle mask
    fig.add_trace(go.Scatter(
        x=x_ventricles, y=y_ventricles,
        mode='markers',
        marker=dict(color='lightblue', size=1.5, opacity=0.6),
        name='Ventricle Mask',
        showlegend=False
    ), row=1, col=1)

    # Ventricle measurement line
    fig.add_trace(go.Scatter(
        x=[ventricle_segment_info['x1'], ventricle_segment_info['x2']],
        y=[ventricle_segment_info['y'], ventricle_segment_info['y']],
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=8, color='red'),
        name=f'Ventricle Width: {ventricle_segment_info["width"]} px',
        showlegend=True
    ), row=1, col=1)

    # Add skull mask
    fig.add_trace(go.Scatter(
        x=x_skull, y=y_skull,
        mode='markers',
        marker=dict(color='lightgray', size=1.5, opacity=0.4),
        name='Brain Mask',
        showlegend=False
    ), row=1, col=2)

    # Skull measurement line
    fig.add_trace(go.Scatter(
        x=[skull_segment_info['x1'], skull_segment_info['x2']],
        y=[skull_segment_info['y'], skull_segment_info['y']],
        mode='lines+markers',
        line=dict(color='green', width=4),
        marker=dict(size=8, color='green'),
        name=f'Skull Width: {skull_segment_info["width"]} px',
        showlegend=True
    ), row=1, col=2)

    # Calculate Evans Index
    evans_index = ventricle_segment_info['width'] / skull_segment_info['width']

    # Create title with anatomical information
    title_parts = [f'Evans Index Measurement: {evans_index:.3f}']

    if 'search_region' in ventricle_segment_info:
        if ventricle_segment_info['search_region'] == 'frontal_horns':
            title_parts.append('✓ Frontal Horn Region')
        else:
            title_parts.append('⚠️ Full Ventricle Search')

    # Add validation information
    if validation_info:
        if validation_info['is_valid']:
            title_parts.append(f'✓ Valid (Score: {validation_info["confidence_score"]:.2f})')
        else:
            title_parts.append('⚠️ Validation Issues')

    title_text = ' | '.join(title_parts)

    # Layout adjustments
    fig.update_layout(
        title_text=title_text,
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    # Match axes and invert Y (medical convention)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, autorange='reversed', row=1, col=1)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, autorange='reversed', row=1, col=2)
    fig.update_xaxes(title_text='X (Left-Right)', row=1, col=1)
    fig.update_xaxes(title_text='X (Left-Right)', row=1, col=2)
    fig.update_yaxes(title_text='Y (Anterior-Posterior)', row=1, col=1)
    fig.update_yaxes(title_text='Y (Anterior-Posterior)', row=1, col=2)

    # Add anatomical information as annotations
    annotations = []

    # Add slice information
    annotations.append(dict(
        x=0.25, y=0.02,
        xref='paper', yref='paper',
        text=f'Slice Z={slice_index} | Occupancy: {ventricle_segment_info.get("occupancy", 0):.2f}',
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)'
    ))

    # Add validation warnings if any
    if validation_info and validation_info['warnings']:
        warning_text = 'Warnings: ' + '; '.join(validation_info['warnings'][:2])  # Show first 2 warnings
        annotations.append(dict(
            x=0.75, y=0.02,
            xref='paper', yref='paper',
            text=warning_text,
            showarrow=False,
            font=dict(size=10, color='orange'),
            bgcolor='rgba(255,255,255,0.8)'
        ))

    fig.update_layout(annotations=annotations)

    if save_path:
        fig.write_html(save_path)
        print(f"Visualization saved to: {save_path}")

    fig.show()

    return fig

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

def run_hydrocephalus_analysis(ventricle_nii_path, skull_nii_path, z_fixed=None, y_fixed=None,
                             occupancy_threshold=0.9, frontal_horn_only=True,
                             include_validation=True, show_visualization=False):
    """
    Enhanced function to run medically-accurate hydrocephalus analysis using Evans Index.

    Medical Standard: Evans Index should be measured at the frontal horns (anterior horns)
    of the lateral ventricles according to established radiological protocols.

    Parameters:
        ventricle_nii_path (str): path to the ventricle mask
        skull_nii_path (str): path to the skull mask
        z_fixed (int or None): fixed slice index for skull analysis (if None, taken from best ventricle segment)
        y_fixed (int or None): fixed column index for skull analysis (if None, taken from best ventricle segment)
        occupancy_threshold (float): minimum occupancy threshold for ventricles (default: 0.9)
        frontal_horn_only (bool): if True, restrict search to frontal horn region (default: True)
        include_validation (bool): if True, perform anatomical validation (default: True)
        show_visualization (bool): if True, display enhanced visualization (default: False)

    Returns:
        dict with comprehensive analysis results:
            - best_ventricle_segment: ventricle measurement details with anatomical info
            - skull_segment: corresponding skull measurement
            - hydrocephalus_result: Evans Index calculation and interpretation
            - validation: anatomical validation results (if include_validation=True)
            - medical_interpretation: enhanced medical assessment
    """
    # Find best ventricle segment with anatomical constraints
    best_ventricle_segment = find_best_ventricle_segment(
        ventricle_nii_path,
        occupancy_threshold=occupancy_threshold,
        frontal_horn_only=frontal_horn_only
    )

    # Use coordinates from the best ventricle segment if not specified
    if z_fixed is None:
        z_fixed = best_ventricle_segment['z']
    if y_fixed is None:
        y_fixed = best_ventricle_segment['y']

    # Find corresponding skull segment
    skull_segment = find_skull_segment(skull_nii_path, z_fixed)

    # Calculate Evans Index and basic interpretation
    hydrocephalus_result = check_hydrocephalus(
        ventricles_width=best_ventricle_segment['width'],
        skull_width=skull_segment['width']
    )

    # Prepare comprehensive results
    results = {
        'best_ventricle_segment': best_ventricle_segment,
        'skull_segment': skull_segment,
        'hydrocephalus_result': hydrocephalus_result
    }

    # Add anatomical validation if requested
    if include_validation:
        validation = validate_frontal_horn_anatomy(best_ventricle_segment, skull_segment)
        results['validation'] = validation

        # Enhanced medical interpretation
        evans_index = best_ventricle_segment['width'] / skull_segment['width']
        medical_interpretation = {
            'evans_index': round(evans_index, 3),
            'measurement_quality': 'Good' if validation['is_valid'] else 'Needs Review',
            'anatomical_compliance': 'Standard' if best_ventricle_segment.get('search_region') == 'frontal_horns' else 'Non-standard',
            'confidence_score': validation['confidence_score'],
            'clinical_significance': _get_clinical_significance(evans_index, validation),
            'recommendations': validation['recommendations']
        }
        results['medical_interpretation'] = medical_interpretation

    # Show visualization if requested
    if show_visualization:
        validation_info = results.get('validation', None)
        visualize_evans_index_measurement(
            ventricle_nii_path, skull_nii_path,
            best_ventricle_segment, skull_segment,
            validation_info=validation_info
        )

    return results

def _get_clinical_significance(evans_index, validation):
    """Helper function to determine clinical significance of Evans Index."""
    if evans_index <= 0.30:
        base_significance = "Normal - hydrocephalus unlikely"
    elif evans_index <= 0.40:
        base_significance = "Borderline - may indicate early hydrocephalus"
    elif evans_index <= 0.50:
        base_significance = "Elevated - suggests hydrocephalus"
    else:
        base_significance = "Significantly elevated - strong indication of hydrocephalus"

    if not validation['is_valid']:
        base_significance += " (measurement quality issues - manual review recommended)"

    return base_significance

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
        # "ventricles_combined": "lateral_ventricles_combined.nii.gz"
        "ventricles_combined": os.path.join("brain_structures", "ventricle.nii.gz")
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

def ei_estimation_pipeline(base: str, verbosity: bool = False, frontal_horn_only: bool = True,
                          include_validation: bool = True, show_visualization: bool = False):
    """
    Enhanced pipeline for medically-accurate Evans Index calculation and hydrocephalus analysis.

    This pipeline follows medical standards for Evans Index measurement by:
    1. Properly aligning brain images using FLIRT transformation matrices
    2. Restricting measurements to frontal horn regions (anterior 40% of ventricles)
    3. Performing anatomical validation of measurement locations
    4. Providing comprehensive medical interpretation

    Medical Standards Applied:
    - Evans Index measured at frontal horns of lateral ventricles
    - Axial plane measurement following radiological protocols
    - Quality validation to ensure measurement accuracy

    Args:
        base (str): Path to the base directory containing required files:
                   - aligning.mat (FLIRT transformation matrix)
                   - brain_mask_combined.nii.gz (skull mask)
                   - brain_structures/ventricle.nii.gz (ventricle mask)
        verbosity (bool): If True, prints detailed information. Default is False.
        frontal_horn_only (bool): If True, restricts search to frontal horn region (recommended). Default is True.
        include_validation (bool): If True, performs anatomical validation. Default is True.
        show_visualization (bool): If True, displays enhanced measurement visualization. Default is False.

    Returns:
        dict: Comprehensive analysis results including:
            - best_ventricle_segment: ventricle measurement with anatomical info
            - skull_segment: skull measurement details
            - hydrocephalus_result: basic Evans Index calculation
            - validation: anatomical validation results (if enabled)
            - medical_interpretation: enhanced clinical assessment (if validation enabled)

    Raises:
        FileNotFoundError: If required files are missing from the base directory
        ValueError: If no suitable ventricle segments are found

    Example:
        >>> results = ei_estimation_pipeline('/path/to/patient/data/',
        ...                                 verbosity=True,
        ...                                 show_visualization=True)
        >>> print(f"Evans Index: {results['medical_interpretation']['evans_index']}")
        >>> print(f"Clinical significance: {results['medical_interpretation']['clinical_significance']}")
    """
    # Check for required files and get paths
    paths_dict, is_success = check_and_get_paths(base)
    if not is_success:
        raise FileNotFoundError("Required files not found in folder: " + base)

    if verbosity:
        print("✅ Found all required files:")
        for key, path in paths_dict.items():
            if not key.startswith('aligned'):  # Don't print aligned paths (they're generated)
                print(f"   {key}: {path}")

    # Extract Euler angles from the FLIRT matrix
    mat_file = paths_dict['mat_file']
    roll, pitch, yaw = extract_euler_angles_from_flirt_mat(mat_file)
    if verbosity:
        print(f"\n📐 Image alignment angles:")
        print(f"   Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")

    # Apply Euler rotation to brain mask
    if verbosity:
        print("\n🔄 Applying rotational alignment...")
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

    # Run enhanced hydrocephalus analysis
    if verbosity:
        print(f"\n🧠 Running Evans Index analysis:")
        print(f"   Frontal horn restriction: {'Yes' if frontal_horn_only else 'No'}")
        print(f"   Validation enabled: {'Yes' if include_validation else 'No'}")

    results = run_hydrocephalus_analysis(
        ventricle_nii_path=paths_dict['aligned_ventricles'],
        skull_nii_path=paths_dict['aligned_brain'],
        occupancy_threshold=0.9,
        frontal_horn_only=frontal_horn_only,
        include_validation=include_validation,
        show_visualization=show_visualization
    )

    if verbosity:
        print(f"\n📊 Results Summary:")
        if include_validation and 'medical_interpretation' in results:
            mi = results['medical_interpretation']
            print(f"   Evans Index: {mi['evans_index']}")
            print(f"   Measurement Quality: {mi['measurement_quality']}")
            print(f"   Anatomical Compliance: {mi['anatomical_compliance']}")
            print(f"   Clinical Significance: {mi['clinical_significance']}")
            if mi['recommendations']:
                print(f"   Recommendations: {'; '.join(mi['recommendations'])}")
        else:
            print(f"   Evans Index: {results['best_ventricle_segment']['width'] / results['skull_segment']['width']:.3f}")
            print(f"   Basic result: {results['hydrocephalus_result']}")

        print(f"\n📍 Measurement Location:")
        vs = results['best_ventricle_segment']
        print(f"   Slice Z={vs['z']}, Y={vs['y']}")
        print(f"   Ventricle width: {vs['width']} pixels")
        print(f"   Skull width: {results['skull_segment']['width']} pixels")
        print(f"   Occupancy: {vs.get('occupancy', 'N/A'):.2f}")
        if 'search_region' in vs:
            print(f"   Search region: {vs['search_region']}")

    return results

if __name__ == "__main__":

    base = "/Users/maratorozaliev/Desktop/MindScope/data/000518240B_155448/"
    results = ei_estimation_pipeline(base)

    print("Best ventricle segment:")
    print(results['best_ventricle_segment'])
    print("\nSkull segment:")
    print(results['skull_segment'])
    print("\nHydrocephalus check result:")
    print(results['hydrocephalus_result'])