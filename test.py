import os
import subprocess

input_dir = "/Users/maratorozaliev/Desktop/cases_in_nii"
output_base_dir = "/Users/maratorozaliev/Desktop/brains"

files = [
    "head_ct_5.nii.gz",
    "head_ct_3.nii.gz",
    "head_ct_4.nii.gz",
    "head_ct_2.nii.gz"
]

for file in files:
    base_name = os.path.splitext(os.path.splitext(file)[0])[0]
    output_folder = os.path.join(output_base_dir, f"output_{base_name}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    input_path = os.path.join(input_dir, file)
    
    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", output_folder,
        "--task", "brain_structures",
        "--device", "cpu"
    ]


    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Processed {file} successfully.")
    else:
        print(f"Error processing {file}:\n{result.stderr}")
