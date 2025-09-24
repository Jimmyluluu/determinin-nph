#!/usr/bin/env python3
"""
Evans Index 分析主模組
"""
import os
import json
from typing import Dict, List, Optional
from utils import find_available_datasets, check_prelabeled_data_paths, load_hydrocephalus_reference, calculate_evans_index, validate_results_against_reference
from image_processing import create_brain_mask_from_original, merge_left_right_ventricles, find_frontal_horns_segment, find_skull_segment
from visualization import generate_evans_slice_screenshot, generate_markdown_report


def run_prelabeled_evans_analysis(base_path: str, dataset_name: str, occupancy_threshold: float = 0.7, generate_screenshots: bool = True, screenshot_output_dir: str = "evans_slices") -> Optional[Dict]:
    """
    使用標記好的資料執行 Evans Index 分析

    Parameters:
        base_path (str): 資料基礎路徑
        dataset_name (str): 資料集名稱
        occupancy_threshold (float): 佔有率閾值，預設0.7
        generate_screenshots (bool): 是否生成可視化截圖，預設True
        screenshot_output_dir (str): 截圖輸出目錄，預設"evans_slices"
    """

    # 檢查檔案路徑
    paths, success = check_prelabeled_data_paths(base_path, dataset_name)
    if not success:
        return None

    # 準備腦室遮罩 - 統一使用左右腦室合併
    ventricle_mask_path = os.path.join(paths["dataset_path"], "merged_lateral_ventricles.nii.gz")
    if not os.path.exists(ventricle_mask_path):
        success = merge_left_right_ventricles(
            paths["ventricle_left"],
            paths["ventricle_right"],
            ventricle_mask_path,
            dataset_name,
            paths["original"]
        )
        if not success:
            return None

    # 準備腦部遮罩 - 統一使用原始影像
    brain_mask_path = os.path.join(paths["dataset_path"], "brain_mask_from_original.nii.gz")
    if not os.path.exists(brain_mask_path):
        success = create_brain_mask_from_original(paths["original"], brain_mask_path)
        if not success:
            print(f"❌ 無法建立腦部遮罩，跳過 {dataset_name}")
            return None

    try:
        # 找出側腦室前角位置 - Evans Index 標準測量點
        print("🔍 尋找側腦室前角測量段...")
        ventricle_segment = find_frontal_horns_segment(ventricle_mask_path, dataset_name, occupancy_threshold=occupancy_threshold)

        if ventricle_segment['width'] == 0:
            print(f"❌ 在 {dataset_name} 中找不到有效的腦室段")
            return None

        # 使用相同的 z, y 座標找出對應的顱骨段
        skull_segment = find_skull_segment(brain_mask_path, ventricle_segment['z'], ventricle_segment['y'])

        # 計算 Evans Index
        evans_results = calculate_evans_index(
            ventricle_segment['width'],
            skull_segment['width']
        )

        # 整合結果
        results = {
            "dataset": dataset_name,
            "ventricle_segment": ventricle_segment,
            "skull_segment": skull_segment,
            "evans_analysis": evans_results,
            "files_used": {
                "original": paths["original"],
                "ventricles": ventricle_mask_path,
                "brain_mask": brain_mask_path,
                "ventricle_left": paths.get("ventricle_left", "N/A"),
                "ventricle_right": paths.get("ventricle_right", "N/A"),
                "needs_merge": paths["needs_merge"]
            }
        }

        # 生成可視化截圖
        if generate_screenshots:
            try:
                print(f"🖼️ 正在為 {dataset_name} 生成可視化截圖...")
                success = generate_evans_slice_screenshot(
                    case_name=dataset_name,
                    original_path=paths["original"],
                    ventricle_path=ventricle_mask_path,
                    brain_mask_path=brain_mask_path,
                    ventricle_coords=ventricle_segment,
                    skull_coords=skull_segment,
                    output_dir=screenshot_output_dir
                )

                if success:
                    screenshot_path = os.path.join(screenshot_output_dir, f'{dataset_name}_evans_slice.png')
                    results["screenshot_path"] = screenshot_path
                    print(f"✅ 截圖已生成: {screenshot_path}")
                else:
                    print(f"⚠️ {dataset_name}: 截圖生成失敗")

            except Exception as screenshot_error:
                print(f"❌ {dataset_name}: 截圖生成出錯 - {str(screenshot_error)}")

        return results

    except Exception as e:
        return None


def batch_analyze_prelabeled_data(base_path: str, datasets: Optional[List[str]] = None, occupancy_threshold: float = 0.7, generate_screenshots: bool = True, screenshot_output_dir: str = "evans_slices") -> Dict:
    """
    批次分析多個標記資料集

    Parameters:
        base_path (str): 資料基礎路徑
        datasets (Optional[List[str]]): 指定要分析的資料集，若為None則分析所有可用資料集
        occupancy_threshold (float): 佔有率閾值，預設0.7
        generate_screenshots (bool): 是否生成可視化截圖，預設True
        screenshot_output_dir (str): 截圖輸出目錄，預設"evans_slices"
    """
    if datasets is None:
        datasets = find_available_datasets(base_path)

    results = {}
    failed_cases = []
    successful = 0
    failed = 0

    for dataset in datasets:
        result = run_prelabeled_evans_analysis(base_path, dataset, occupancy_threshold, generate_screenshots, screenshot_output_dir)
        if result:
            results[dataset] = result
            successful += 1
        else:
            failed_cases.append(dataset)
            failed += 1

    print(f"\n分析完成: 成功 {successful}, 失敗 {failed}")
    if failed_cases:
        print(f"失敗案例: {failed_cases}")

    # 將失敗案例加入結果中以便報告顯示
    results["_failed_cases"] = failed_cases

    # 顯示統計摘要
    if results and len(results) > 1:  # 確保有實際結果（不只是 _failed_cases）
        # 排除特殊鍵
        actual_results = {k: v for k, v in results.items() if not k.startswith('_')}
        if actual_results:
            import numpy as np
            evans_indices = [r["evans_analysis"]["evans_index"] for r in actual_results.values()]
            avg_evans = np.mean(evans_indices)
            high_risk_count = sum(1 for r in actual_results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "高")
            medium_risk_count = sum(1 for r in actual_results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "中")

            print(f"\n📊 統計摘要:")
            print(f"   平均 Evans Index: {avg_evans:.4f}")
            print(f"   腦室擴大案例: {high_risk_count}/{len(actual_results)} ({high_risk_count/len(actual_results)*100:.1f}%)")
            print(f"   可能/早期擴大: {medium_risk_count}/{len(actual_results)} ({medium_risk_count/len(actual_results)*100:.1f}%)")

    return results


if __name__ == "__main__":
    # 設定標記資料的路徑
    LABELED_DATA_PATH = "/Volumes/Kuro醬の1TSSD/標記好的資料"

    if not os.path.exists(LABELED_DATA_PATH):
        print(f"找不到資料路徑: {LABELED_DATA_PATH}")
        exit(1)

    # 找出所有可用的資料集
    available_datasets = find_available_datasets(LABELED_DATA_PATH)
    print(f"發現 {len(available_datasets)} 個資料集")

    # 設定佔有率閾值 - 只有佔有率 >= 此值的腦室段才會被考慮
    OCCUPANCY_THRESHOLD = 0.7  # 可以根據需要調整此值 (0.0-1.0)

    # 設定截圖輸出目錄
    SCREENSHOT_OUTPUT_DIR = "evans_slices"
    os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)

    # 執行批次分析（包含自動截圖生成）
    batch_results = batch_analyze_prelabeled_data(
        LABELED_DATA_PATH,
        occupancy_threshold=OCCUPANCY_THRESHOLD,
        generate_screenshots=True,
        screenshot_output_dir=SCREENSHOT_OUTPUT_DIR
    )

    # 載入已知水腦症參考清單
    known_hydrocephalus = load_hydrocephalus_reference()
    if known_hydrocephalus:
        print(f"\n載入已知水腦症案例: {len(known_hydrocephalus)} 個")

        # 驗證結果
        validation = validate_results_against_reference(batch_results, known_hydrocephalus)

        print(f"\n驗證結果:")
        print(f"  總體準確率: {validation['accuracy']:.1%}")
        print(f"  水腦症正確識別: {validation['hydrocephalus_correctly_identified']}/{validation['known_hydrocephalus_count']}")
        print(f"  正常案例正確識別: {validation['normal_correctly_identified']}")

        if validation['false_negatives']:
            print(f"  漏報案例: {len(validation['false_negatives'])} 個")

        if validation['false_positives']:
            print(f"  誤報案例: {len(validation['false_positives'])} 個")

        if validation['not_analyzed']:
            print(f"  未分析案例: {len(validation['not_analyzed'])} 個")

    # 建立結果資料夾
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    # 保存結果到 JSON 檔案
    output_file = os.path.join(result_dir, "prelabeled_evans_analysis_results.json")

    # 將驗證結果加入 JSON
    final_results = {
        "analysis_results": batch_results,
        "validation": validation if known_hydrocephalus else None,
        "known_hydrocephalus_cases": known_hydrocephalus
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"JSON 結果已保存: {output_file}")

    # 產生 Markdown 報告
    md_file = os.path.join(result_dir, "prelabeled_evans_analysis_report.md")
    generate_markdown_report(batch_results, md_file, validation if known_hydrocephalus else None)
    print(f"Markdown 報告已保存: {md_file}")

    print(f"\n📁 所有結果檔案已保存在 {result_dir}/ 資料夾")
    print(f"🖼️ 可視化截圖已保存在 {SCREENSHOT_OUTPUT_DIR}/ 資料夾")