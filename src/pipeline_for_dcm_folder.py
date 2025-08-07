import os
from src.segmentator.utils import generate_token_from_filename
from src.segmentator.convert_dicom import run_dcm2niix
from src.segmentator.run_brain_segmentation import run_totalsegmentator
from src.segmentator.utils import combine_brain_structures, combine_lateral_ventricles
from src.segmentator.run_aligner_flirt import run_flirt
import json

def run_pipeline_dcm_to_data_folder(dcm_folder_path, totalseg_token, project_root=".", data_folder_name="data"):
    """
    Creates a new directory under project_root/data/{file_name_token} for a DICOM folder input.

    Parameters:
        dcm_folder_path (str): Path to the folder containing DICOM files
        project_root (str): Root path of the project
        data_folder_name (str): Folder name where to save the output inside the project root

    Returns:
        str: Path to the created output directory
    """
    # Generate a unique token based on folder name
    token = generate_token_from_filename(dcm_folder_path)
    # Construct full path: project_root/data/token
    output_dir = os.path.join(project_root, data_folder_name, token)
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Created output directory: {output_dir}")
    

    # Step 1: DICOM to NIfTI
    run_dcm2niix(input_dir=dcm_folder_path, output_dir=output_dir, filename_prefix='head_ct')
    ct_path = os.path.join(output_dir, "head_ct.nii.gz")


    # Step 2: Create subfolders
    brain_structures_dir = os.path.join(output_dir, "brain_structures")
    ventricles_dir = os.path.join(output_dir, "ventricles")
    os.makedirs(brain_structures_dir, exist_ok=True)
    os.makedirs(ventricles_dir, exist_ok=True)

    # Step 3: Run TotalSegmentator
    run_totalsegmentator(ct_path=ct_path, output_path=brain_structures_dir, token=totalseg_token, task="brain_structures")
    run_totalsegmentator(ct_path=ct_path, output_path=ventricles_dir, token=totalseg_token, task="ventricle_parts")

   # Step 4: Combine structure masks with explicit output paths
    brain_mask_path = combine_brain_structures(
        input_dir=brain_structures_dir,
        output_path=os.path.join(output_dir, "brain_mask_combined.nii.gz")
    )

    ventricle_mask_path = combine_lateral_ventricles(
        input_dir=ventricles_dir,
        output_path=os.path.join(output_dir, "lateral_ventricles_combined.nii.gz")
    )

    # Step 4: Run FLIRT
    ref_path = os.path.join(project_root, data_folder_name, "brain_blueprint", "MNI152_T1_1mm_brain.nii")
    mat_path = os.path.join(output_dir, "aligning.mat")

    run_flirt(input_path=brain_mask_path, ref_path=ref_path, output_mat=mat_path)

    # Step 5: Save all paths as JSON
    paths = {
        "output_dir": output_dir,
        "ct_path": ct_path,
        "brain_mask_path": brain_mask_path,
        "ventricle_mask_path": ventricle_mask_path,
        "ref_path": ref_path,
        "flirt_matrix_path": mat_path
    }

    json_path = os.path.join(output_dir, "file_paths.json")
    with open(json_path, "w") as f:
        json.dump(paths, f, indent=4)

    print(f"[INFO] Saved all file paths to: {json_path}")
    return paths

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    segmentator_token = os.getenv("TOTALSEGMENTATOR_TOKEN")

    paths = run_pipeline_dcm_to_data_folder(
        dcm_folder_path="/Users/maratorozaliev/Desktop/cases/2-000067886E/",
        totalseg_token=segmentator_token,
        project_root="/Users/maratorozaliev/Desktop/MindScope"
    )