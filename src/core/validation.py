#!/usr/bin/env python3
"""
臨床驗證與 Evans Index 計算模組
"""
import os
import json
from typing import Dict, List


def calculate_evans_index(ventricle_width: float, skull_width: float) -> Dict:
    """
    計算 Evans Index 並提供臨床解釋

    Parameters:
        ventricle_width (float): 腦室寬度
        skull_width (float): 顱骨寬度

    Returns:
        Dict: Evans Index 結果，包含臨床意義
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


def load_hydrocephalus_reference() -> List[str]:
    """
    載入已知水腦症案例的參考清單

    Returns:
        List[str]: 已知水腦症案例列表
    """
    try:
        # 嘗試多個可能的路徑
        possible_paths = [
            "hydrocephalus_reference.json",
            "result/hydrocephalus_reference.json",
            "../result/hydrocephalus_reference.json"
        ]

        for reference_file in possible_paths:
            if os.path.exists(reference_file):
                with open(reference_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("hydrocephalus_cases", {}).get("cases", [])
        return []
    except Exception:
        return []


def validate_results_against_reference(results: Dict, known_hydrocephalus: List[str]) -> Dict:
    """
    驗證分析結果與已知臨床診斷的一致性

    Parameters:
        results (Dict): 分析結果字典
        known_hydrocephalus (List[str]): 已知水腦症案例列表

    Returns:
        Dict: 驗證結果統計
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
