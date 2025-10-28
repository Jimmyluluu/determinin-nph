#!/usr/bin/env python3
"""
資料載入與檔案路徑管理模組
"""
import os
import glob
from typing import Dict, List, Tuple


def find_available_datasets(base_path: str) -> List[str]:
    """
    找出所有可用的標記資料集（包括 data_X 和病例號格式）

    Parameters:
        base_path (str): 資料基礎路徑

    Returns:
        List[str]: 資料集名稱列表
    """
    datasets = []

    # 找 data_X 格式的資料夾
    data_pattern = os.path.join(base_path, "data_*")
    for data_dir in glob.glob(data_pattern):
        if os.path.isdir(data_dir):
            dataset_name = os.path.basename(data_dir)
            if dataset_name not in ["data_16_not_ok"]:  # 排除問題資料
                datasets.append(dataset_name)

    # 找病例號格式的資料夾（以數字開頭的資料夾）
    case_pattern = os.path.join(base_path, "0*")
    for case_dir in glob.glob(case_pattern):
        if os.path.isdir(case_dir):
            case_name = os.path.basename(case_dir)
            datasets.append(case_name)

    datasets.sort()
    return datasets


def check_prelabeled_data_paths(base_path: str, dataset_name: str) -> Tuple[Dict[str, str], bool]:
    """
    檢查標記資料的路徑是否存在（支援兩種格式）

    Parameters:
        base_path (str): 資料基礎路徑
        dataset_name (str): 資料集名稱

    Returns:
        Tuple[Dict[str, str], bool]: (路徑字典, 是否成功)
    """
    dataset_path = os.path.join(base_path, dataset_name)

    # 判斷是 data_X 格式還是病例號格式
    if dataset_name.startswith("data_"):
        # data_X 格式
        dataset_num = dataset_name.split("_")[1]
        paths = {
            "dataset_path": dataset_path,
            "original": os.path.join(dataset_path, f"original_{dataset_num}.nii.gz"),
            "ventricles": os.path.join(dataset_path, f"mask_Ventricles_{dataset_num}.nii.gz"),
            "ventricle_left": os.path.join(dataset_path, f"mask_Ventricle_L_{dataset_num}.nii.gz"),
            "ventricle_right": os.path.join(dataset_path, f"mask_Ventricle_R_{dataset_num}.nii.gz"),
            "csf": os.path.join(dataset_path, f"mask_CSF_{dataset_num}.nii.gz"),
        }
    else:
        # 病例號格式
        paths = {
            "dataset_path": dataset_path,
            "original": os.path.join(dataset_path, "original.nii.gz"),
            "ventricles": os.path.join(dataset_path, "Ventricles.nii.gz"),
            "ventricle_left": os.path.join(dataset_path, "Ventricle_L.nii.gz"),
            "ventricle_right": os.path.join(dataset_path, "Ventricle_R.nii.gz"),
            "csf": os.path.join(dataset_path, "CSF.nii.gz"),
        }

    # 檢查哪些檔案存在
    existing_paths = {}
    missing_files = []

    for key, path in paths.items():
        if os.path.exists(path):
            existing_paths[key] = path
        else:
            missing_files.append(os.path.basename(path))

    # 檢查必要檔案
    # Evans Index 必須使用左右側腦室，不能使用包含四腦室和三腦室的 Ventricles
    if "ventricle_left" in existing_paths and "ventricle_right" in existing_paths:
        existing_paths["needs_merge"] = True
    else:
        # 檢查是否有 Ventricles 檔案但沒有左右分離檔案
        if "ventricles" in existing_paths:
            print(f"⚠️ {dataset_name}: 只有 Ventricles 檔案，無法進行 Evans Index 分析（需要左右腦室分離）")
        else:
            print(f"❌ {dataset_name}: 缺少左右腦室檔案")
        return existing_paths, False

    # 檢查 original 檔案
    if "original" not in existing_paths:
        print(f"❌ {dataset_name}: 缺少原始影像檔案")
        return existing_paths, False

    return existing_paths, True
