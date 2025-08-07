# MindScope

Automatic tool for detecting Evans Index and analyzing brain structures from CT scans.

---

## Overview

MindScope is a Python 3.10-based pipeline for automatic brain structure segmentation, image alignment, and calculation of Evans Index and Ventricle-to-Brain Ratio (VBR) from brain CT scans.

The program takes a folder containing DICOM files of head CT, processes them using TotalSegmentator, aligns images with FSL's flirt tool, and produces volumetric statistics along with visualizations.

---

## Requirements

- Python 3.10
- [TotalSegmentator](https://github.com/TotalSegmentator/TotalSegmentator) (requires access token)
- [FSL (flirt)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) for image alignment
- [dcm2niix](https://github.com/rordenlab/dcm2niix) for DICOM to NIfTI conversion
- Python dependencies listed in `requirements.txt`

---

## Installation

1. Clone this repository:

   ```bash
   git clone <repository_url>
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

## Usage

Run the main pipeline script with the path to your DICOM folder:

```bash
python src/pipeline_for_dcm_folder.py
```

After processing completes, open the Jupyter notebook for visualization and detailed analysis:

```bash
notebooks/demo_evans_index.ipynb
```

---

## Outputs

- Volumetric statistics of brain structures, including Evans Index and VBR
- Segmentation files (NIfTI format)
- Alignment parameters (rotation angles)
- Visualizations and reports available in the notebook

---

## TODO

- Add detailed usage examples and command line options
- Document output files structure and formats
- Add scripts for automated environment setup
- Expand troubleshooting and configuration instructions

---
