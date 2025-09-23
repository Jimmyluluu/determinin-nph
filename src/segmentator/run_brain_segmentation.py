import subprocess
import os
from dotenv import load_dotenv
load_dotenv()
segmentator_token = os.getenv("TOTALSEGMENTATOR_TOKEN")

def run_totalsegmentator(ct_path: str, output_path: str, token: str = None, task: str = "brain_structures"):
    """
    Runs TotalSegmentator for segmentation of brain structures or ventricles.

    Parameters:
        ct_path (str): path to the CT file (e.g., 'head_ct.nii.gz')
        output_path (str): path to the folder for saving results
        token (str): authentication token (optional, not needed for standard tasks)
        task (str): segmentation task, either 'brain_structures' or 'ventricles'
    """
    if task not in ["brain_structures", "ventricle_parts"]:
        raise ValueError("For hydrocephalus detection, the 'task' parameter must be either 'brain_structures' or 'ventricle_parts'")

    # 使用相對於項目根目錄的路徑
    totalseg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "env", "bin", "TotalSegmentator")

    command = [
        totalseg_path,
        "-i", ct_path,
        "-o", output_path,
        "--task", task,
        "--device", "cpu",
        "--statistics"
    ]

    # 只有在有 token 時才添加 license 參數
    if token is not None:
        command.extend(["-l", token])
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Segmentation completed successfully.")
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing command:")
        print(e.stderr)

if __name__ == "__main__":
    # Пример использования:
    run_totalsegmentator("head_ct.nii.gz", "output_brain_seg", token=segmentator_token)
