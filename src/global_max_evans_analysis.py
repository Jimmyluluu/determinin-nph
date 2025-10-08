#!/usr/bin/env python3
"""
全域最大值 Evans Index 分析模組
腦室最大寬度和顱骨最大寬度可以來自不同的切片
"""
import os
import json
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional
from utils import calculate_evans_index
from image_processing import find_skull_segment
from visualization import generate_global_max_screenshot


def find_global_max_measurements(nii_path: str, brain_mask_path: str, dataset_name: str = "",
                                 max_reasonable_width: int = 200, occupancy_threshold: float = 0.6) -> Dict:
    """
    在前角範圍內找出腦室和顱骨的全域最大值

    Parameters:
        nii_path (str): 腦室遮罩路徑
        brain_mask_path (str): 腦部遮罩路徑
        dataset_name (str): 資料集名稱
        max_reasonable_width (int): 最大合理寬度
        occupancy_threshold (float): 佔有率閾值

    Returns:
        Dict: 包含腦室和顱骨的全域最大測量結果
    """
    img = nib.load(nii_path)
    mask_data = img.get_fdata()
    binary = (mask_data > 0).astype(np.uint8)

    X, Y, Z = binary.shape

    # 檢查遮罩是否有內容
    total_pixels = np.count_nonzero(binary)
    if total_pixels == 0:
        print(f"❌ {dataset_name}: 腦室遮罩完全為空")
        return {}

    # 找出腦室的 Z 軸範圍
    z_coords = []
    for z in range(Z):
        if np.count_nonzero(binary[:, :, z]) > 0:
            z_coords.append(z)

    if not z_coords:
        return {}

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

    print(f"🔍 {dataset_name}: 在前角範圍 Z={target_z_start}-{target_z_end} 中尋找全域最大值...")

    # 初始化全域最大值
    global_max_ventricle = {'width': 0, 'z': None, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}
    global_max_skull = {'width': 0, 'z': None, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}

    # 逐切片掃描，尋找全域最大值
    for z in range(target_z_start, min(target_z_end + 1, Z)):
        slice_ = binary[:, :, z]

        if np.count_nonzero(slice_) == 0:
            continue

        # 找該切片的最佳腦室測量
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

            # 更新全域最大腦室測量
            if occupancy >= occupancy_threshold and width > global_max_ventricle['width']:
                global_max_ventricle.update({
                    'width': int(width),
                    'z': int(z),
                    'y': int(y),
                    'x1': int(x1),
                    'x2': int(x2),
                    'occupancy': float(occupancy)
                })

        # 如果該切片有有效的腦室測量，也嘗試找顱骨測量
        if np.count_nonzero(slice_) > 0:
            try:
                # 使用任意一個 y 座標來觸發 find_skull_segment
                # find_skull_segment 會自動在整個 Y 軸找最大顱骨寬度
                y_for_skull = y_search_range[len(y_search_range)//2]  # 使用中間的 y
                skull_segment = find_skull_segment(brain_mask_path, z, y_for_skull)

                # 更新全域最大顱骨測量
                if skull_segment['width'] > global_max_skull['width']:
                    global_max_skull = skull_segment

            except Exception as e:
                continue

    # 檢查是否找到有效的測量
    if global_max_ventricle['width'] == 0:
        print(f"❌ {dataset_name}: 沒有找到有效的腦室測量")
        return {}

    if global_max_skull['width'] == 0:
        print(f"❌ {dataset_name}: 沒有找到有效的顱骨測量")
        return {}

    # 計算 Evans Index
    evans_results = calculate_evans_index(
        global_max_ventricle['width'],
        global_max_skull['width']
    )

    result = {
        'ventricle_max': global_max_ventricle,
        'skull_max': global_max_skull,
        'evans_analysis': evans_results,
        'same_slice': global_max_ventricle['z'] == global_max_skull['z']
    }

    print(f"✅ {dataset_name}: 腦室最大={global_max_ventricle['width']} (Z={global_max_ventricle['z']}), "
          f"顱骨最大={global_max_skull['width']} (Z={global_max_skull['z']}), "
          f"Evans Index={evans_results['evans_index']:.4f}")

    return result


def run_global_max_analysis_for_case(case_name: str, case_paths: Dict,
                                     occupancy_threshold: float = 0.6,
                                     output_base_dir: str = "result/global_max") -> Optional[Dict]:
    """
    為單個案例執行全域最大值分析
    """
    try:
        print(f"\n🔍 開始全域最大值分析: {case_name}")

        # 尋找全域最大值
        global_max_result = find_global_max_measurements(
            case_paths['ventricles'],
            case_paths['brain_mask'],
            case_name,
            occupancy_threshold=occupancy_threshold
        )

        if not global_max_result:
            print(f"❌ {case_name}: 全域最大值分析失敗")
            return None

        # 創建輸出目錄
        case_output_dir = os.path.join(output_base_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)

        # 生成視覺化截圖
        screenshot_success = generate_global_max_screenshot(
            case_name=case_name,
            original_path=case_paths['original'],
            ventricle_path=case_paths['ventricles'],
            brain_mask_path=case_paths['brain_mask'],
            ventricle_max=global_max_result['ventricle_max'],
            skull_max=global_max_result['skull_max'],
            evans_index=global_max_result['evans_analysis']['evans_index'],
            output_dir=case_output_dir
        )

        # 保存結果到 JSON
        json_path = os.path.join(case_output_dir, "global_max_data.json")
        output_data = {
            "case_name": case_name,
            "method": "global_max",
            "ventricle_max": global_max_result['ventricle_max'],
            "skull_max": global_max_result['skull_max'],
            "evans_analysis": global_max_result['evans_analysis'],
            "same_slice": global_max_result['same_slice'],
            "screenshot_generated": screenshot_success
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        summary = {
            "case_name": case_name,
            "ventricle_max_z": global_max_result['ventricle_max']['z'],
            "skull_max_z": global_max_result['skull_max']['z'],
            "same_slice": global_max_result['same_slice'],
            "evans_index": global_max_result['evans_analysis']['evans_index'],
            "screenshot_generated": screenshot_success,
            "data_file": json_path
        }

        print(f"✅ {case_name}: 全域最大值分析完成")
        return summary

    except Exception as e:
        print(f"❌ {case_name}: 全域最大值分析失敗 - {str(e)}")
        return None


def generate_global_max_summary_report(all_case_summaries: List[Dict], output_path: str):
    """
    生成全域最大值分析摘要報告 (參考原始報告格式)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 全域最大值 Evans Index 分析報告\n\n")
        f.write(f"📅 分析時間: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 過濾出成功分析的案例
        successful_cases = [s for s in all_case_summaries if s is not None]
        failed_count = len(all_case_summaries) - len(successful_cases)

        if not successful_cases:
            f.write("❌ 沒有成功分析的資料集\n")
            if failed_count > 0:
                f.write(f"\n### ❌ 分析失敗案例 ({failed_count} 個)\n")
            return

        # 統計摘要
        evans_indices = [s['evans_index'] for s in successful_cases]
        avg_evans = np.mean(evans_indices)

        normal_cases = [s for s in successful_cases if s['evans_index'] <= 0.25]
        mild_cases = [s for s in successful_cases if 0.25 < s['evans_index'] <= 0.30]
        high_cases = [s for s in successful_cases if s['evans_index'] > 0.30]

        f.write("## 📊 統計摘要\n\n")
        f.write(f"- **總共分析案例**: {len(successful_cases)} 個\n")
        if failed_count > 0:
            f.write(f"- **分析失敗案例**: {failed_count} 個\n")
        f.write(f"- **平均 Evans Index**: {avg_evans:.4f}\n")
        f.write(f"- **正常範圍 (≤ 0.25)**: {len(normal_cases)}/{len(successful_cases)} ({len(normal_cases)/len(successful_cases)*100:.1f}%)\n")
        f.write(f"- **可能/早期擴大 (0.25-0.30)**: {len(mild_cases)}/{len(successful_cases)} ({len(mild_cases)/len(successful_cases)*100:.1f}%)\n")
        f.write(f"- **腦室擴大 (> 0.30)**: {len(high_cases)}/{len(successful_cases)} ({len(high_cases)/len(successful_cases)*100:.1f}%)\n\n")

        # 統計腦室和顱骨在不同切片的案例數
        diff_slice_cases = [s for s in successful_cases if not s['same_slice']]
        f.write(f"- **腦室與顱骨在不同切片的案例**: {len(diff_slice_cases)}/{len(successful_cases)} ({len(diff_slice_cases)/len(successful_cases)*100:.1f}%)\n\n")

        # 正常範圍案例
        f.write("## 🟢 正常範圍案例\n\n")
        if normal_cases:
            f.write("| 案例 | Evans Index | 腦室最大 Z | 顱骨最大 Z | 同切片? |\n")
            f.write("|------|-------------|------------|------------|----------|\n")
            for case in sorted(normal_cases, key=lambda x: x['case_name']):
                same_slice = "✓" if case['same_slice'] else "✗"
                f.write(f"| {case['case_name']} | {case['evans_index']:.4f} | {case['ventricle_max_z']} | {case['skull_max_z']} | {same_slice} |\n")
        else:
            f.write("沒有正常範圍的案例\n")

        # 可能/早期擴大案例
        f.write("\n## 🟡 可能/早期擴大案例\n\n")
        if mild_cases:
            f.write("| 案例 | Evans Index | 腦室最大 Z | 顱骨最大 Z | 同切片? | 臨床意義 |\n")
            f.write("|------|-------------|------------|------------|----------|----------|\n")
            for case in sorted(mild_cases, key=lambda x: x['case_name']):
                same_slice = "✓" if case['same_slice'] else "✗"
                f.write(f"| {case['case_name']} | {case['evans_index']:.4f} | {case['ventricle_max_z']} | {case['skull_max_z']} | {same_slice} | 可能或早期腦室擴大 (0.25-0.30) |\n")
        else:
            f.write("沒有可能/早期擴大案例\n")

        # 腦室擴大案例
        f.write("\n## 🔴 腦室擴大案例\n\n")
        if high_cases:
            f.write("| 案例 | Evans Index | 腦室最大 Z | 顱骨最大 Z | 同切片? | 臨床意義 |\n")
            f.write("|------|-------------|------------|------------|----------|----------|\n")
            for case in sorted(high_cases, key=lambda x: x['case_name']):
                same_slice = "✓" if case['same_slice'] else "✗"
                f.write(f"| {case['case_name']} | {case['evans_index']:.4f} | {case['ventricle_max_z']} | {case['skull_max_z']} | {same_slice} | 腦室擴大 (> 0.30) |\n")
        else:
            f.write("沒有腦室擴大案例\n")

        # 已知水腦症案例驗證
        known_hydrocephalus_cases = [
            "000235496D", "000206288G", "000152785B",
            "000137208D", "000096384I", "000087554H"
        ]

        # 建立案例字典方便查詢
        case_dict = {case['case_name']: case for case in successful_cases}

        f.write("\n## 🔍 已知水腦症案例驗證\n\n")

        # 計算驗證統計
        analyzed_known_cases = [c for c in known_hydrocephalus_cases if c in case_dict]
        correct_identified = sum(1 for c in analyzed_known_cases if case_dict[c]['evans_index'] > 0.30)

        if analyzed_known_cases:
            accuracy = correct_identified / len(analyzed_known_cases)
            f.write(f"- **已知水腦症案例數**: {len(known_hydrocephalus_cases)} 個\n")
            f.write(f"- **成功分析案例**: {len(analyzed_known_cases)} 個\n")
            f.write(f"- **正確識別 (> 0.30)**: {correct_identified}/{len(analyzed_known_cases)} ({accuracy*100:.1f}%)\n")
            f.write(f"- **漏報案例**: {len(analyzed_known_cases) - correct_identified} 個\n\n")

            # 詳細表格
            f.write("### 📋 已知水腦症案例預測狀況\n\n")
            f.write("| 案例 | Evans Index | 預測結果 | 實際狀況 | 狀態 |\n")
            f.write("|------|-------------|----------|----------|------|\n")

            for case_name in known_hydrocephalus_cases:
                if case_name in case_dict:
                    case = case_dict[case_name]
                    evans_idx = case['evans_index']

                    # 判斷預測結果
                    if evans_idx <= 0.25:
                        predicted = "低"
                    elif evans_idx <= 0.30:
                        predicted = "中"
                    else:
                        predicted = "高"

                    # 判斷是否正確
                    status = "✅ 正確" if evans_idx > 0.30 else "❌ 漏報"

                    f.write(f"| {case_name} | {evans_idx:.4f} | {predicted} 風險 | 有水腦症 | {status} |\n")
                else:
                    f.write(f"| {case_name} | - | 未分析 | 有水腦症 | ❌ 未分析 |\n")
        else:
            f.write("無已知水腦症案例參與本次分析\n")

        # 說明
        f.write("\n## 📖 說明\n\n")
        f.write("- **Evans Index**: 腦室寬度與顱骨寬度的比值\n")
        f.write("- **正常範圍**: ≤ 0.25\n")
        f.write("- **可能/早期腦室擴大**: 0.25-0.30\n")
        f.write("- **腦室擴大**: > 0.30\n")
        f.write("- **測量方法 (全域最大值法)**: \n")
        f.write("  - 腦室最大寬度: 在前角範圍內所有切片中找到的最大值\n")
        f.write("  - 顱骨最大寬度: 在前角範圍內所有切片中找到的最大值 (每個切片掃描整個 Y 軸)\n")
        f.write("  - **特點**: 腦室和顱骨的最大值可能來自不同的切片\n\n")

        if failed_count > 0:
            f.write("## ❌ 分析失敗案例\n\n")
            f.write(f"共 {failed_count} 個案例分析失敗\n")

    print(f"📄 報告已生成: {output_path}")
