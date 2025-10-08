#!/usr/bin/env python3
"""
逐切片 Evans Index 分析模組
在確定的前角範圍內為每個切片計算 Evans Index
"""
import os
import json
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional
from utils import calculate_evans_index
from image_processing import find_skull_segment
from visualization import generate_evans_slice_screenshot


def analyze_slices_in_frontal_horn_range(nii_path: str, brain_mask_path: str, dataset_name: str = "",
                                        max_reasonable_width: int = 200, occupancy_threshold: float = 0.6) -> List[Dict]:
    """
    在前角範圍內分析所有切片的 Evans Index

    Parameters:
        nii_path (str): 腦室遮罩路徑
        brain_mask_path (str): 腦部遮罩路徑
        dataset_name (str): 資料集名稱
        max_reasonable_width (int): 最大合理寬度
        occupancy_threshold (float): 佔有率閾值

    Returns:
        List[Dict]: 所有有效切片的測量結果
    """
    img = nib.load(nii_path)
    mask_data = img.get_fdata()
    binary = (mask_data > 0).astype(np.uint8)

    X, Y, Z = binary.shape
    slice_results = []

    # 檢查遮罩是否有內容
    total_pixels = np.count_nonzero(binary)
    if total_pixels == 0:
        print(f"❌ {dataset_name}: 腦室遮罩完全為空")
        return slice_results

    # 找出腦室的 Z 軸範圍（重用原邏輯）
    z_coords = []
    for z in range(Z):
        if np.count_nonzero(binary[:, :, z]) > 0:
            z_coords.append(z)

    if not z_coords:
        return slice_results

    z_min, z_max = min(z_coords), max(z_coords)
    z_range = z_max - z_min

    # 前角區域：Z 軸前部 (z/3 到 z)
    if z_range > 0:
        target_z_start = int(z_min + z_range / 3)
        target_z_end = z_max
    else:
        target_z_start = target_z_end = z_min

    # 根據資料來源決定 Y 軸搜索範圍
    y_mid = Y // 2
    is_data_series = dataset_name.startswith('data_')

    if is_data_series:
        y_search_range = range(0, y_mid)
        search_description = "下半部 (data 系列)"
    else:
        y_search_range = range(y_mid, Y)
        search_description = "上半部 (編號系列)"

    print(f"🔍 {dataset_name}: 在前角範圍 Z={target_z_start}-{target_z_end} 中分析所有切片...")

    # 逐切片分析
    for z in range(target_z_start, min(target_z_end + 1, Z)):
        slice_ = binary[:, :, z]

        if np.count_nonzero(slice_) == 0:
            continue

        # 找該切片的最佳腦室測量
        best_ventricle = {'width': 0, 'z': z, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}

        for y in y_search_range:
            col = slice_[:, y]
            xs = np.where(col > 0)[0]

            if xs.size < 2:
                continue

            x1, x2 = xs.min(), xs.max()
            width = x2 - x1

            # 檢查寬度合理性
            if width > max_reasonable_width or width < 5:
                continue

            # 檢查佔有率
            occupancy = col[x1:x2+1].sum() / (width + 1) if width > 0 else 0

            # 更新最佳測量
            if occupancy >= occupancy_threshold and width > best_ventricle['width']:
                best_ventricle.update({
                    'width': int(width),
                    'z': int(z),
                    'y': int(y),
                    'x1': int(x1),
                    'x2': int(x2),
                    'occupancy': float(occupancy)
                })

        # 如果找到有效的腦室測量，尋找對應的顱骨測量
        if best_ventricle['width'] > 0:
            try:
                skull_segment = find_skull_segment(brain_mask_path, z, best_ventricle['y'])

                # 計算 Evans Index
                evans_results = calculate_evans_index(
                    best_ventricle['width'],
                    skull_segment['width']
                )

                # 記錄切片結果
                slice_result = {
                    'slice_z': z,
                    'ventricle_segment': best_ventricle,
                    'skull_segment': skull_segment,
                    'evans_analysis': evans_results
                }

                slice_results.append(slice_result)

            except Exception as e:
                print(f"⚠️ {dataset_name} Z={z}: 顱骨測量失敗 - {str(e)}")
                continue

    print(f"✅ {dataset_name}: 在範圍內找到 {len(slice_results)} 個有效切片")
    return slice_results


def generate_slice_screenshots_for_case(case_name: str, original_path: str, ventricle_path: str,
                                       brain_mask_path: str, slice_results: List[Dict],
                                       output_base_dir: str) -> List[str]:
    """
    為案例的所有切片生成截圖

    Returns:
        List[str]: 生成的截圖路徑列表
    """
    case_output_dir = os.path.join(output_base_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    screenshot_paths = []

    for slice_data in slice_results:
        z_slice = slice_data['slice_z']
        evans_index = slice_data['evans_analysis']['evans_index']

        # 生成包含 Evans Index 的檔名
        screenshot_filename = f"slice_{z_slice:03d}_evans_{evans_index:.4f}.png"
        screenshot_path = os.path.join(case_output_dir, screenshot_filename)

        # 使用現有的截圖生成函數，但指定特定的輸出路徑
        try:
            success = generate_evans_slice_screenshot(
                case_name=f"{case_name}_slice_{z_slice}",
                original_path=original_path,
                ventricle_path=ventricle_path,
                brain_mask_path=brain_mask_path,
                ventricle_coords=slice_data['ventricle_segment'],
                skull_coords=slice_data['skull_segment'],
                output_dir=case_output_dir
            )

            if success:
                # 重命名檔案為我們想要的格式
                old_path = os.path.join(case_output_dir, f"{case_name}_slice_{z_slice}_evans_slice.png")
                if os.path.exists(old_path):
                    os.rename(old_path, screenshot_path)
                    screenshot_paths.append(screenshot_path)

        except Exception as e:
            print(f"❌ {case_name} 切片 {z_slice} 截圖生成失敗: {str(e)}")
            continue

    return screenshot_paths


def save_case_slice_data(case_name: str, slice_results: List[Dict], output_base_dir: str) -> str:
    """
    保存案例的所有切片數據到 JSON 檔案（排除該案例內部的離群切片）
    """
    case_output_dir = os.path.join(output_base_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    json_path = os.path.join(case_output_dir, "slices_data.json")

    if not slice_results:
        data = {
            "case_name": case_name,
            "total_slices": 0,
            "slice_range": {"min_z": None, "max_z": None},
            "evans_index_stats": {"min": None, "max": None, "mean": None, "std": None},
            "slice_details": []
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return json_path

    # 提取所有 Evans Index 值
    evans_indices = [s['evans_analysis']['evans_index'] for s in slice_results]

    # 計算原始統計
    original_stats = {
        "min": min(evans_indices),
        "max": max(evans_indices),
        "mean": np.mean(evans_indices),
        "std": np.std(evans_indices),
        "count": len(evans_indices)
    }

    # 使用 IQR 方法檢測該案例內部的離群切片
    filtered_indices = evans_indices
    outlier_slices = []

    if len(evans_indices) > 4:  # 需要足夠的切片數才進行離群值檢測
        q1 = np.percentile(evans_indices, 25)
        q3 = np.percentile(evans_indices, 75)
        iqr = q3 - q1

        if iqr > 0:  # 避免所有值都相同的情況
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # 分離正常切片和離群切片
            filtered_indices = []
            for i, slice_result in enumerate(slice_results):
                evans_idx = slice_result['evans_analysis']['evans_index']
                if lower_bound <= evans_idx <= upper_bound:
                    filtered_indices.append(evans_idx)
                else:
                    outlier_info = {
                        "slice_z": slice_result['slice_z'],
                        "evans_index": evans_idx,
                        "reason": "below_threshold" if evans_idx < lower_bound else "above_threshold"
                    }
                    outlier_slices.append(outlier_info)

    # 計算排除離群值後的統計
    filtered_stats = {
        "min": min(filtered_indices) if filtered_indices else None,
        "max": max(filtered_indices) if filtered_indices else None,
        "mean": np.mean(filtered_indices) if filtered_indices else None,
        "std": np.std(filtered_indices) if filtered_indices else None,
        "count": len(filtered_indices)
    }

    # 整理數據
    data = {
        "case_name": case_name,
        "total_slices": len(slice_results),
        "slice_range": {
            "min_z": min(s['slice_z'] for s in slice_results),
            "max_z": max(s['slice_z'] for s in slice_results)
        },
        "evans_index_stats": filtered_stats,  # 使用排除離群值後的統計
        "evans_index_stats_original": original_stats,  # 保留原始統計
        "outlier_detection": {
            "outlier_slices": outlier_slices,
            "outlier_count": len(outlier_slices),
            "detection_method": "IQR_1.5" if len(evans_indices) > 4 else "disabled_insufficient_data"
        },
        "slice_details": slice_results
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"📊 {case_name}: 總切片 {len(slice_results)} 個，排除離群值 {len(outlier_slices)} 個，有效切片 {len(filtered_indices)} 個")

    return json_path


def run_slice_by_slice_analysis_for_case(case_name: str, case_paths: Dict,
                                        occupancy_threshold: float = 0.6,
                                        output_base_dir: str = "result/detailed_slices") -> Optional[Dict]:
    """
    為單個案例執行逐切片分析
    """
    try:
        print(f"\n🔍 開始逐切片分析: {case_name}")

        # 分析所有切片
        slice_results = analyze_slices_in_frontal_horn_range(
            case_paths['ventricles'],
            case_paths['brain_mask'],
            case_name,
            occupancy_threshold=occupancy_threshold
        )

        if not slice_results:
            print(f"❌ {case_name}: 沒有找到有效的切片")
            return None

        # 生成截圖
        screenshot_paths = generate_slice_screenshots_for_case(
            case_name,
            case_paths['original'],
            case_paths['ventricles'],
            case_paths['brain_mask'],
            slice_results,
            output_base_dir
        )

        # 保存數據
        json_path = save_case_slice_data(case_name, slice_results, output_base_dir)

        # 讀取保存的數據以獲取排除離群值後的統計
        with open(json_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        summary = {
            "case_name": case_name,
            "total_slices": len(slice_results),
            "effective_slices": saved_data['evans_index_stats']['count'],  # 排除離群值後的有效切片數
            "outlier_slices": saved_data['outlier_detection']['outlier_count'],  # 離群值切片數
            "slice_range": {
                "min_z": min(s['slice_z'] for s in slice_results),
                "max_z": max(s['slice_z'] for s in slice_results)
            },
            "evans_index_stats": saved_data['evans_index_stats'],  # 排除離群值後的統計
            "evans_index_stats_original": saved_data['evans_index_stats_original'],  # 原始統計
            "outlier_detection": saved_data['outlier_detection'],
            "screenshots_generated": len(screenshot_paths),
            "data_file": json_path
        }

        print(f"✅ {case_name}: 完成 {len(slice_results)} 個切片分析，生成 {len(screenshot_paths)} 張截圖")
        return summary

    except Exception as e:
        print(f"❌ {case_name}: 逐切片分析失敗 - {str(e)}")
        return None


def generate_detailed_summary_report(all_case_summaries: List[Dict], output_path: str):
    """
    生成所有案例的逐切片分析詳細摘要報告
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 逐切片 Evans Index 分析詳細報告\n\n")
        f.write(f"📅 分析時間: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 過濾出成功分析的案例
        successful_cases = [s for s in all_case_summaries if s is not None]
        failed_count = len(all_case_summaries) - len(successful_cases)

        f.write("## 📊 整體統計\n\n")
        f.write(f"- **總共處理案例**: {len(all_case_summaries)} 個\n")
        f.write(f"- **成功分析案例**: {len(successful_cases)} 個\n")
        if failed_count > 0:
            f.write(f"- **分析失敗案例**: {failed_count} 個\n")

        if successful_cases:
            total_slices = sum(s['total_slices'] for s in successful_cases)
            effective_slices = sum(s['effective_slices'] for s in successful_cases)
            total_outlier_slices = sum(s['outlier_slices'] for s in successful_cases)
            total_screenshots = sum(s['screenshots_generated'] for s in successful_cases)

            f.write(f"- **總共分析切片**: {total_slices} 個\n")
            f.write(f"- **有效切片數** (排除個案內離群值): {effective_slices} 個\n")
            f.write(f"- **個案內離群切片**: {total_outlier_slices} 個\n")
            f.write(f"- **生成截圖數量**: {total_screenshots} 張\n")

            # Evans Index 整體統計（使用各案例排除離群值後的平均值）
            all_evans_means = []
            all_evans_mins = []
            all_evans_maxs = []

            for case in successful_cases:
                if case['evans_index_stats']['mean'] is not None:
                    all_evans_means.append(case['evans_index_stats']['mean'])
                    all_evans_mins.append(case['evans_index_stats']['min'])
                    all_evans_maxs.append(case['evans_index_stats']['max'])

            if all_evans_means:
                overall_mean = np.mean(all_evans_means)
                overall_std = np.std(all_evans_means)

                f.write(f"- **Evans Index 範圍**: {min(all_evans_mins):.4f} - {max(all_evans_maxs):.4f}\n")
                f.write(f"- **整體平均 Evans Index** (各案例均已排除內部離群值): {overall_mean:.4f} ± {overall_std:.4f}\n\n")

        # 個案詳細表格
        f.write("## 📋 個案分析詳情\n\n")
        if successful_cases:
            f.write("| 案例 | 總切片 | 有效切片 | 離群切片 | Z範圍 | Evans Index (排除離群值) | 風險評估 | 截圖數 |\n")
            f.write("|------|--------|----------|----------|-------|--------------------------|----------|--------|\n")

            for case in sorted(successful_cases, key=lambda x: x['case_name']):
                case_name = case['case_name']
                total_slices = case['total_slices']
                effective_slices = case['effective_slices']
                outlier_slices = case['outlier_slices']
                z_range = f"{case['slice_range']['min_z']}-{case['slice_range']['max_z']}"

                stats = case['evans_index_stats']
                if stats['mean'] is not None:
                    evans_display = f"{stats['mean']:.4f} ({stats['min']:.4f}-{stats['max']:.4f})"
                    mean_evans = stats['mean']
                else:
                    evans_display = "無有效數據"
                    mean_evans = 0

                # 根據平均值判斷風險
                if mean_evans <= 0.25:
                    risk = "🟢 正常"
                elif mean_evans <= 0.30:
                    risk = "🟡 可能擴大"
                else:
                    risk = "🔴 腦室擴大"

                # 標記有離群切片的案例
                if outlier_slices > 0:
                    risk += f" (排除{outlier_slices}個)"

                screenshot_count = case['screenshots_generated']

                f.write(f"| {case_name} | {total_slices} | {effective_slices} | {outlier_slices} | {z_range} | {evans_display} | {risk} | {screenshot_count} |\n")
        else:
            f.write("沒有成功分析的案例\n")

        # 風險分類統計
        if successful_cases:
            f.write("\n## 🎯 風險分類統計\n\n")

            # 使用排除內部離群值後的統計進行分類
            normal_cases = []
            mild_cases = []
            high_cases = []

            for case in successful_cases:
                if case['evans_index_stats']['mean'] is not None:
                    mean_evans = case['evans_index_stats']['mean']
                    if mean_evans <= 0.25:
                        normal_cases.append(case)
                    elif mean_evans <= 0.30:
                        mild_cases.append(case)
                    else:
                        high_cases.append(case)

            total = len(successful_cases)

            f.write(f"- **正常範圍 (≤ 0.25)**: {len(normal_cases)}/{total} ({len(normal_cases)/total*100:.1f}%)\n")
            f.write(f"- **可能擴大 (0.25-0.30)**: {len(mild_cases)}/{total} ({len(mild_cases)/total*100:.1f}%)\n")
            f.write(f"- **腦室擴大 (> 0.30)**: {len(high_cases)}/{total} ({len(high_cases)/total*100:.1f}%)\n\n")

            # 內部離群值統計
            cases_with_outliers = [s for s in successful_cases if s['outlier_slices'] > 0]
            f.write(f"- **含有內部離群切片的案例**: {len(cases_with_outliers)}/{total} ({len(cases_with_outliers)/total*100:.1f}%)\n\n")

            # 高風險案例詳情
            if high_cases:
                f.write("### 🔴 腦室擴大案例詳情\n\n")
                for case in sorted(high_cases, key=lambda x: x['evans_index_stats']['mean'], reverse=True):
                    stats = case['evans_index_stats']
                    original_stats = case['evans_index_stats_original']
                    outlier_count = case['outlier_slices']

                    f.write(f"- **{case['case_name']}**: ")
                    f.write(f"平均 {stats['mean']:.4f} (排除{outlier_count}個離群切片後), ")
                    f.write(f"範圍 {stats['min']:.4f}-{stats['max']:.4f}, ")
                    f.write(f"標準差 {stats['std']:.4f}")
                    if outlier_count > 0:
                        f.write(f" [原始平均: {original_stats['mean']:.4f}]")
                    f.write("\n")

        # 檔案結構說明
        f.write("\n## 📁 檔案結構說明\n\n")
        f.write("```\n")
        f.write("result/detailed_slices/\n")
        f.write("├── 案例名稱/\n")
        f.write("│   ├── slice_XXX_evans_Y.YYYY.png  # 切片截圖（檔名含Evans Index）\n")
        f.write("│   └── slices_data.json            # 該案例所有切片的詳細數據\n")
        f.write("└── detailed_summary.md             # 本報告檔案\n")
        f.write("```\n\n")

        # 說明
        f.write("## 📖 說明\n\n")
        f.write("- 本分析在前角範圍內對每個有效切片進行 Evans Index 計算\n")
        f.write("- 只包含佔有率 ≥ 0.6 且寬度合理的切片\n")
        f.write("- **離群值處理**: 對每個案例內部的切片使用 IQR 1.5倍方法檢測並排除離群值\n")
        f.write("- **統計基準**: 所有統計數據（平均值、範圍等）均基於排除內部離群切片後的有效切片\n")
        f.write("- 截圖檔名格式: `slice_[Z座標]_evans_[Evans Index值].png`\n")
        f.write("- Evans Index 正常範圍: ≤ 0.25；可能擴大: 0.25-0.30；腦室擴大: > 0.30\n")
        f.write("- 風險評估欄位中的 \"(排除N個)\" 表示該案例排除了 N 個內部離群切片\n")
        f.write("- 每個案例的詳細數據（包含原始統計和離群切片資訊）保存在對應資料夾的 `slices_data.json` 中\n")