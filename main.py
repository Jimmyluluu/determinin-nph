# import os
# from src.segmentator.utils import generate_token_from_filename
# from src.segmentator.convert_dicom import run_dcm2niix
# from src.segmentator.run_brain_segmentation import run_totalsegmentator
# from src.segmentator.utils import combine_brain_structures, combine_lateral_ventricles
# from src.segmentator.run_aligner_flirt import run_flirt
# import json
# import csv
# from dotenv import load_dotenv

# if __name__ == "__main__":

#     ventricle_mask_path = combine_lateral_ventricles(
#         input_dir="/Users/maratorozaliev/Desktop/MindScope/data/002680008I_153956/ventricles",
#         output_path=os.path.join("/Users/maratorozaliev/Desktop/MindScope/data/002680008I_153956/", "lateral_ventricles_combined.nii.gz")
#     )