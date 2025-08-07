# MindScope

Automatic tool for detecting Evans Index and analyzing brain structures from CT scans.

---

## Overview

MindScope is a Python 3.10-based pipeline for automatic brain structure segmentation, image alignment, and calculation of Evans Index and Ventricle-to-Brain Ratio (VBR) from brain CT scans.

The program takes a folder containing DICOM files of head CT, processes them using TotalSegmentator, aligns images with FSL's flirt tool, and produces volumetric statistics along with visualizations.

---

## Requirements

- Python 3.10
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (requires access token)
- [FSL (flirt)](https://fsl.fmrib.ox.ac.uk/fsldownloads) for image alignment
- [dcm2niix](https://github.com/rordenlab/dcm2niix) for DICOM to NIfTI conversion
```bash
brew install dcm2niix
```
- Python dependencies listed in `requirements.txt`

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/maratNeSlaiv/MindScope.git
   cd MindScope
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install and configure external tools:
   - Install TotalSegmentator and obtain your token.
   - Install FSL to get access to `flirt`.
   - Install dcm2niix for DICOM conversion.

---

## Configuration

Create a `.env` file in the project root directory with your TotalSegmentator token:

```env
TOTALSEGMENTATOR_TOKEN=<your_totalsegmentator_token>
```

---

## 🚀 Usage

To process your own DICOM CT scans folder:

```bash
python src/pipeline_for_dcm_folder.py <- specify your /absolute/path/to/dicom/folder
```

---

## 🧪 Sample Case

This section demonstrates a sample use case of the pipeline.

### 📥 Download Sample DICOM Case

You can download the sample DICOM folder [here](https://drive.google.com/drive/folders/1XmbWorwfuCjpnybmHxFOpMrm1TaLT1gh?usp=share_link).

### ⚙️ Run Command

Run the main pipeline script with the path to your DICOM folder:

```bash
python src/pipeline_for_dcm_folder.py
```

If you get "ModuleNotFoundError" you can try specifying project root directly with:
```bash
export PYTHONPATH=/absolute/path/to/MindScope
```

After processing completes, open the Jupyter notebook for visualization and detailed analysis:

```bash
notebooks/demo_evans_index.ipynb
```
You need to specify the "base" parameter with your own path to MindScope/data/generated_token/.

---

### 📈 Output

After running all scripts (including demo_evans_index.ipynb), you will find:
| File / Folder             | Description                                                                         |
|---------------------------|-------------------------------------------------------------------------------------|
| `segmentation_*.nii.gz`   | Segmentation masks for brain regions                                                |
| `statistics.json`         | Volumes and calculated metrics like VBR = Ventricle-to-Brain Ratio, EI = Evans Index|
| `aligning.mat`            | Alignment rotation angles for pitch correction                                      |
| `brain_mask_aligned.nii`  | Aligned brain mask                                                                  |

---

### 🗂️ Folder Structure

```
project-root/
├── notebooks/
│   └── demo_evans_index.ipynb
├── src/
│   └── pipeline_for_dcm_folder.py
├── data/
│   └── generated_token/
│       ├── brain_structures/
│       │   ├── segmentation_1.nii.gz
│       │   ├── segmentation_2.nii.gz
│       │   ├── ...
│       │   └── statistics.json
│       ├── ventricles/
│       │   ├── segmentation_1.nii.gz
│       │   ├── segmentation_2.nii.gz
│       │   ├── ...
│       │   └── statistics.json
│       ├── aligning.mat
│       └── etc...
├── .env  ← specify your TOTALSEGMENTATOR_TOKEN here
```

---

You can use this sample case to verify the pipeline setup and output interpretation.
## TODO

- Add scripts for automated environment setup
- Expand troubleshooting and configuration instructions

---
