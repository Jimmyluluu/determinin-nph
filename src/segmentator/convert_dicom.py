import subprocess

def run_dcm2niix(input_dir, output_dir, filename_prefix= "head_ct"):
    # Prepare the dcm2niix command to convert DICOM to NIfTI format
    command = [
        "dcm2niix",
        "-z", "y",              # Compress output to .nii.gz
        "-o", output_dir,       # Output directory
        "-f", filename_prefix,  # Output filename prefix
        input_dir               # Input directory containing DICOM files
    ]
    try:
        # Run the command and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Command executed successfully!")
        print("stdout:", result.stdout)   # Print standard output
        print("stderr:", result.stderr)   # Print standard error (warnings, etc.)
    except subprocess.CalledProcessError as e:
        # Handle error if the command fails
        print("Error occurred while executing the command:")
        print(e)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)

if __name__ == "__main__":
    # Example call to the function
    run_dcm2niix("./1-000325562H", "./output", filename_prefix= "head_ct")
