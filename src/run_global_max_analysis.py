#!/usr/bin/env python3
"""
執行全域最大值 Evans Index 分析的主程式
腦室最大寬度和顱骨最大寬度可以來自不同的切片
"""
import os
from utils import find_available_datasets, check_prelabeled_data_paths
from image_processing import create_brain_mask_from_original, merge_left_right_ventricles
from global_max_evans_analysis import (
    run_global_max_analysis_for_case,
    generate_global_max_summary_report
)

# 設定參數
BASE_PATH = "/Volumes/Kuro醬の1TSSD/標記好的資料"
OCCUPANCY_THRESHOLD = 0.6
OUTPUT_DIR = "result/global_max"


def main():
    print("=" * 80)
    print("全域最大值 Evans Index 分析")
    print("=" * 80)
    print(f"\n📂 資料路徑: {BASE_PATH}")
    print(f"🔢 佔有率閾值: {OCCUPANCY_THRESHOLD}")
    print(f"📁 輸出目錄: {OUTPUT_DIR}\n")

    # 檢查資料路徑
    if not os.path.exists(BASE_PATH):
        print(f"❌ 找不到資料路徑: {BASE_PATH}")
        return

    # 找出所有可用的資料集
    available_datasets = find_available_datasets(BASE_PATH)
    print(f"📊 發現 {len(available_datasets)} 個資料集\n")

    # 創建輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_summaries = []
    success_count = 0
    fail_count = 0

    # 逐案例分析
    for i, dataset_name in enumerate(available_datasets, 1):
        print(f"\n[{i}/{len(available_datasets)}] 處理: {dataset_name}")
        print("-" * 80)

        # 檢查檔案路徑
        paths, path_success = check_prelabeled_data_paths(BASE_PATH, dataset_name)
        if not path_success:
            all_summaries.append(None)
            fail_count += 1
            continue

        # 準備腦室遮罩
        ventricle_mask_path = os.path.join(paths["dataset_path"], "merged_lateral_ventricles.nii.gz")
        if not os.path.exists(ventricle_mask_path):
            merge_success = merge_left_right_ventricles(
                paths["ventricle_left"],
                paths["ventricle_right"],
                ventricle_mask_path,
                dataset_name,
                paths["original"]
            )
            if not merge_success:
                all_summaries.append(None)
                fail_count += 1
                continue

        # 準備腦部遮罩
        brain_mask_path = os.path.join(paths["dataset_path"], "brain_mask_from_original.nii.gz")
        if not os.path.exists(brain_mask_path):
            brain_success = create_brain_mask_from_original(paths["original"], brain_mask_path)
            if not brain_success:
                print(f"❌ 無法建立腦部遮罩，跳過 {dataset_name}")
                all_summaries.append(None)
                fail_count += 1
                continue

        # 組裝案例路徑
        case_paths = {
            'original': paths["original"],
            'ventricles': ventricle_mask_path,
            'brain_mask': brain_mask_path
        }

        # 執行分析
        summary = run_global_max_analysis_for_case(
            case_name=dataset_name,
            case_paths=case_paths,
            occupancy_threshold=OCCUPANCY_THRESHOLD,
            output_base_dir=OUTPUT_DIR
        )

        all_summaries.append(summary)
        if summary:
            success_count += 1
        else:
            fail_count += 1

    # 生成摘要報告
    print("\n" + "=" * 80)
    print("生成摘要報告...")
    print("=" * 80)

    report_path = os.path.join(OUTPUT_DIR, "global_max_summary.md")
    generate_global_max_summary_report(all_summaries, report_path)

    # 最終統計
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"✅ 成功: {success_count} 個案例")
    print(f"❌ 失敗: {fail_count} 個案例")
    print(f"📄 摘要報告: {report_path}")
    print(f"📁 輸出目錄: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
