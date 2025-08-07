import os
from datetime import datetime
import nibabel as nib
import numpy as np
from typing import Optional

def generate_token_from_filename(file_path, time_format="%H%M%S", max_name_length=12):
    """
    Generates a short, unique token based on a file name and current time.

    Parameters:
        file_path (str): Full path to the input file
        time_format (str): Format for the time stamp (e.g., "%Y%m%d_%H%M")
        max_name_length (int): Max length for the base name part

    Returns:
        str: Token like "headct_153045"
    """
    base = os.path.basename(file_path)         # e.g., head_ct.nii.gz
    name, _ = os.path.splitext(base)           # remove extension → head_ct
    name = name[:max_name_length]              # truncate if needed
    time_part = datetime.now().strftime(time_format)
    return f"{name}_{time_part}"


def combine_brain_structures(input_dir: str, output_path: Optional[str] = None) -> str:
    """
    Combines selected brain structure masks from the given directory into a single binary mask.

    Parameters:
        input_dir (str): Path to the folder containing brain structure .nii.gz files.
        output_path (Optional[str]): Output path for the combined mask. If None, a default name will be used.

    Returns:
        str: Path to the saved combined mask.
    """

    # Structures of interest
    structures_of_interest = [
        'central_sulcus.nii.gz',
        'internal_capsule.nii.gz',
        'lentiform_nucleus.nii.gz',
        'cerebellum.nii.gz',
        'occipital_lobe.nii.gz',
        'insular_cortex.nii.gz',
        'temporal_lobe.nii.gz',
        'septum_pellucidum.nii.gz',
        'frontal_lobe.nii.gz',
        'thalamus.nii.gz',
        'caudate_nucleus.nii.gz',
        'parietal_lobe.nii.gz',
        'brainstem.nii.gz'
    ]

    # Find available structure files
    files_to_load = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f in structures_of_interest]

    if not files_to_load:
        raise FileNotFoundError("No target brain structure files found in the directory.")

    # Combine masks
    combined_mask = None
    affine = None
    for filepath in files_to_load:
        img = nib.load(filepath)
        data = img.get_fdata() > 0  # Binarize

        if combined_mask is None:
            combined_mask = data
            affine = img.affine
        else:
            combined_mask = np.logical_or(combined_mask, data)

    # Convert to uint8 and save
    combined_mask = combined_mask.astype(np.uint8)
    brain_mask_img = nib.Nifti1Image(combined_mask, affine)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_dir), "brain_mask_combined.nii.gz")

    nib.save(brain_mask_img, output_path)

    return output_path


def combine_lateral_ventricles(input_dir: str, output_path: Optional[str] = None) -> str:
    """
    Combines selected lateral ventricle masks from the given directory into a single binary mask.

    Parameters:
        input_dir (str): Path to the folder containing ventricle .nii.gz files.
        output_path (Optional[str]): Output path for the combined mask. If None, a default name will be used.

    Returns:
        str: Path to the saved combined mask.
    """

    # Target ventricle parts for 'lateral venctricles'
    ventricles_of_interest = [
        'ventricle_body_left.nii.gz',
        'ventricle_body_right.nii.gz',
        'ventricle_frontal_horn_left.nii.gz',
        'ventricle_frontal_horn_right.nii.gz',
        'ventricle_trigone_left.nii.gz',
        'ventricle_trigone_right.nii.gz',
        'ventricle_occipital_horn_left.nii.gz',
        'ventricle_occipital_horn_right.nii.gz',
        'ventricle_temporal_horn_left.nii.gz',
        'ventricle_temporal_horn_right.nii.gz',
    ]

    # Find available files
    files_to_load = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f in ventricles_of_interest]

    if not files_to_load:
        raise FileNotFoundError("No target ventricle files found in the directory.")

    # Combine masks
    combined_mask = None
    affine = None
    for filepath in files_to_load:
        img = nib.load(filepath)
        data = img.get_fdata() > 0  # binarize

        if combined_mask is None:
            combined_mask = data
            affine = img.affine
        else:
            combined_mask = np.logical_or(combined_mask, data)

    # Convert and save
    combined_mask = combined_mask.astype(np.uint8)
    combined_img = nib.Nifti1Image(combined_mask, affine)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_dir), "lateral_ventricles_combined.nii.gz")

    nib.save(combined_img, output_path)

    return output_path
