#!/usr/bin/env python3
import json
import os
from visualization import generate_evans_slice_screenshot


def main():
    # 讀取分析結果
    results_file = "/Users/lujingyuan/Project/研究相關/MindScope/result/prelabeled_evans_analysis_results.json"

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['analysis_results']
    base_path = "/Volumes/Kuro醬の1TSSD/標記好的資料"
    output_dir = "/Users/lujingyuan/Project/研究相關/MindScope/evans_slices"

    success_count = 0
    total_count = 0

    for case_name, case_data in results.items():
        if case_name.startswith('_'):
            continue

        total_count += 1
        print(f"\nProcessing case: {case_name}")

        try:
            # 取得檔案路徑
            files = case_data['files_used']
            original_path = files['original']
            ventricles_path = files['ventricles']
            brain_mask_path = files['brain_mask']

            # 取得測量座標
            ventricle_coords = case_data['ventricle_segment']
            skull_coords = case_data['skull_segment']

            # 生成截圖
            if generate_evans_slice_screenshot(case_name, original_path, ventricles_path,
                                             brain_mask_path, ventricle_coords, skull_coords, output_dir):
                success_count += 1

        except Exception as e:
            print(f"❌ {case_name}: Processing failed - {str(e)}")

    print(f"\nCompleted! Successfully generated {success_count}/{total_count} screenshots")
    print(f"Screenshots saved in: {output_dir}")

if __name__ == "__main__":
    main()