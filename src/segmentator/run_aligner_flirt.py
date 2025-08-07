import subprocess

def run_flirt(input_path, ref_path, output_mat, dof=6):
    command = [
        "flirt",
        "-in", input_path,
        "-ref", ref_path,
        "-dof", str(dof),
        "-omat", output_mat
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("FLIRT completed successfully.")
        print("STDOUT:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("FLIRT failed.")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

# Example of usage:
if __name__ == '__main__':
    run_flirt(
        input_path="/Users/maratorozaliev/Desktop/brains/brain_mask_combined_4.nii.gz",
        ref_path="/Users/maratorozaliev/Desktop/MNI152_T1_1mm_brain.nii",
        output_mat="/Users/maratorozaliev/Desktop/transform.mat"
    )
