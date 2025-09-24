#!/usr/bin/env python3
"""
共用工具函數模組
"""
import os
import json
import glob
from typing import Dict, List, Tuple


def find_available_datasets(base_path: str) -> List[str]:
    """
    找出所有可用的標記資料集（包括 data_X 和病例號格式）
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


def load_hydrocephalus_reference() -> List[str]:
    """
    載入已知水腦症案例的參考清單
    """
    try:
        reference_file = "hydrocephalus_reference.json"
        if os.path.exists(reference_file):
            with open(reference_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("hydrocephalus_cases", {}).get("cases", [])
        return []
    except Exception:
        return []


def calculate_evans_index(ventricle_width: float, skull_width: float) -> Dict:
    """
    計算 Evans Index 並提供臨床解釋
    """
    if skull_width == 0:
        return {"error": "顱骨寬度不能為零"}

    evans_index = ventricle_width / skull_width

    # 合理性檢查
    warnings = []
    if evans_index > 1.0:
        warnings.append(f"異常: Evans Index > 1.0 ({evans_index:.4f})")
    if ventricle_width > 300:
        warnings.append(f"異常: 腦室寬度過大 ({ventricle_width})")
    if skull_width < 100:
        warnings.append(f"異常: 顱骨寬度過小 ({skull_width})")

    # 更新的臨床分類標準
    if evans_index <= 0.25:
        clinical_significance = "正常範圍 (≤ 0.25)"
        hydrocephalus_risk = "低"
    elif evans_index <= 0.30:
        clinical_significance = "可能或早期腦室擴大 (0.25-0.30)"
        hydrocephalus_risk = "中"
    else:
        clinical_significance = "腦室擴大 (> 0.30)"
        hydrocephalus_risk = "高"

    result = {
        "evans_index": round(float(evans_index), 4),
        "ventricle_width": int(ventricle_width),
        "skull_width": int(skull_width),
        "clinical_significance": clinical_significance,
        "hydrocephalus_risk": hydrocephalus_risk,
        "warnings": warnings if warnings else None
    }

    return result


def validate_results_against_reference(results: Dict, known_hydrocephalus: List[str]) -> Dict:
    """
    驗證分析結果與已知臨床診斷的一致性
    """
    # 排除特殊鍵來計算實際分析的案例數
    actual_results_count = len([k for k in results.keys() if not k.startswith('_')])

    validation = {
        "total_analyzed": actual_results_count,
        "known_hydrocephalus_count": len(known_hydrocephalus),
        "hydrocephalus_correctly_identified": 0,
        "normal_correctly_identified": 0,
        "false_negatives": [],  # 應該是水腦症但被判為正常
        "false_positives": [],  # 應該是正常但被判為水腦症
        "not_analyzed": [],
        "accuracy": 0.0
    }

    total_correct = 0

    # 檢查已知水腦症案例
    for case in known_hydrocephalus:
        if case in results:
            evans_index = results[case]["evans_analysis"]["evans_index"]
            if evans_index > 0.30:  # 只有 > 0.30 才算預測為異常
                validation["hydrocephalus_correctly_identified"] += 1
                total_correct += 1
            else:
                validation["false_negatives"].append({
                    "case": case,
                    "evans_index": evans_index
                })
        else:
            validation["not_analyzed"].append(case)

    # 檢查應該是正常的案例
    for case, result in results.items():
        # 跳過特殊鍵
        if case.startswith('_'):
            continue

        if case not in known_hydrocephalus:  # 應該是正常案例
            evans_index = result["evans_analysis"]["evans_index"]
            if evans_index <= 0.30:  # ≤ 0.30 才算預測為正常
                validation["normal_correctly_identified"] += 1
                total_correct += 1
            else:
                validation["false_positives"].append({
                    "case": case,
                    "evans_index": evans_index
                })

    # 計算準確率
    if validation["total_analyzed"] > 0:
        validation["accuracy"] = total_correct / validation["total_analyzed"]

    return validation