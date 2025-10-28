#!/usr/bin/env python3
"""
MindScope - Evans Index 分析主程式
支援兩種分析模式：全域最大值分析和逐切片分析
"""
import argparse
import sys
import os

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import load_config, AnalysisConfig
from utils import find_available_datasets


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='MindScope - Evans Index 分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
分析模式說明：
  global_max      全域最大值分析 - 在前角範圍內找出腦室和顱骨的全域最大寬度（可來自不同切片）
  slice_by_slice  逐切片分析 - 在前角範圍內為每個切片計算 Evans Index

範例：
  # 使用全域最大值分析
  python main.py --mode global_max

  # 使用逐切片分析
  python main.py --mode slice_by_slice

  # 指定設定檔
  python main.py --mode global_max --config my_config.json

  # 覆蓋特定參數
  python main.py --mode global_max --base-path /path/to/data --threshold 0.7
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['global_max', 'slice_by_slice'],
        required=True,
        help='分析模式'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='設定檔路徑 (JSON 格式)'
    )

    parser.add_argument(
        '--base-path',
        type=str,
        help='資料基礎路徑（覆蓋設定檔）'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help='佔有率閾值（覆蓋設定檔，範圍 0.0-1.0）'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='輸出目錄（覆蓋設定檔）'
    )

    parser.add_argument(
        '--no-screenshots',
        action='store_true',
        help='不生成可視化截圖'
    )

    parser.add_argument(
        '--enable-detailed-slices',
        action='store_true',
        help='啟用逐切片詳細分析（slice_by_slice 模式）'
    )

    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='列出所有可用的資料集並結束'
    )

    parser.add_argument(
        '--save-config',
        type=str,
        help='將目前設定儲存到指定的檔案'
    )

    return parser.parse_args()


def execute_global_max_analysis(config: AnalysisConfig, generate_screenshots: bool = True):
    """執行全域最大值分析"""
    from global_max_evans_analysis import (
        run_global_max_analysis_for_case,
        generate_global_max_summary_report
    )
    from image_processing import create_brain_mask_from_original, merge_left_right_ventricles
    from utils import check_prelabeled_data_paths, load_hydrocephalus_reference, validate_results_against_reference

    print("=" * 80)
    print("全域最大值 Evans Index 分析")
    print("=" * 80)
    print(f"\n📂 資料路徑: {config.base_path}")
    print(f"🔢 佔有率閾值: {config.occupancy_threshold}")
    print(f"📁 輸出目錄: {config.output_dir}\n")

    # 檢查資料路徑
    if not os.path.exists(config.base_path):
        print(f"❌ 找不到資料路徑: {config.base_path}")
        return

    # 找出所有可用的資料集
    available_datasets = find_available_datasets(config.base_path)
    print(f"📊 發現 {len(available_datasets)} 個資料集\n")

    # 載入已知水腦症案例
    known_hydrocephalus = load_hydrocephalus_reference()
    if known_hydrocephalus:
        print(f"📋 載入 {len(known_hydrocephalus)} 個已知水腦症案例\n")

    # 創建輸出目錄
    output_dir = os.path.join(config.output_dir, "global_max")
    cases_dir = os.path.join(output_dir, "cases")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cases_dir, exist_ok=True)

    all_summaries = []
    all_results = {}
    success_count = 0
    fail_count = 0

    # 逐案例分析
    for i, dataset_name in enumerate(available_datasets, 1):
        print(f"\n[{i}/{len(available_datasets)}] 處理: {dataset_name}")
        print("-" * 80)

        # 檢查檔案路徑
        paths, path_success = check_prelabeled_data_paths(config.base_path, dataset_name)
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

        # 執行分析（輸出到 cases 子目錄）
        summary = run_global_max_analysis_for_case(
            case_name=dataset_name,
            case_paths=case_paths,
            occupancy_threshold=config.occupancy_threshold,
            output_base_dir=cases_dir
        )

        all_summaries.append(summary)
        if summary:
            # 從 JSON 檔案讀取完整資料（用於驗證）
            import json
            with open(summary['data_file'], 'r', encoding='utf-8') as f:
                full_data = json.load(f)

            all_results[dataset_name] = {
                "dataset": dataset_name,
                "ventricle_segment": full_data['ventricle_max'],
                "skull_segment": full_data['skull_max'],
                "evans_analysis": full_data['evans_analysis'],
                "files_used": {
                    "original": paths["original"],
                    "ventricles": ventricle_mask_path,
                    "brain_mask": brain_mask_path
                }
            }
            success_count += 1
        else:
            fail_count += 1

    # 驗證結果
    validation = None
    if known_hydrocephalus:
        validation = validate_results_against_reference(all_results, known_hydrocephalus)

    # 生成摘要報告
    print("\n" + "=" * 80)
    print("生成摘要報告...")
    print("=" * 80)

    report_path = os.path.join(output_dir, "analysis_report.md")
    generate_global_max_summary_report(all_summaries, report_path, validation)

    # 儲存完整結果
    import json
    results_file = os.path.join(output_dir, "analysis_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_results': all_results,
            'validation': validation,
            'known_hydrocephalus_cases': known_hydrocephalus,
            'config': config.to_dict()
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ 結果已儲存: {results_file}")

    # 最終統計
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"✅ 成功: {success_count} 個案例")
    print(f"❌ 失敗: {fail_count} 個案例")
    print(f"📄 結果檔案: {results_file}")
    print(f"📄 摘要報告: {report_path}")
    print(f"📁 輸出目錄: {output_dir}")
    print("=" * 80)


def execute_slice_by_slice_analysis(config: AnalysisConfig, generate_screenshots: bool = True):
    """執行逐切片分析（合併單切片和詳細分析）"""
    from slice_by_slice_analysis import run_slice_by_slice_analysis_for_case, generate_detailed_summary_report
    from utils import check_prelabeled_data_paths, load_hydrocephalus_reference, validate_results_against_reference
    from image_processing import create_brain_mask_from_original, merge_left_right_ventricles
    from visualization import generate_evans_slice_screenshot
    import json

    print("=" * 80)
    print("逐切片 Evans Index 分析")
    print("=" * 80)
    print(f"\n📂 資料路徑: {config.base_path}")
    print(f"🔢 佔有率閾值: {config.occupancy_threshold}")
    print(f"📁 輸出目錄: {config.output_dir}\n")

    # 檢查資料路徑
    if not os.path.exists(config.base_path):
        print(f"❌ 找不到資料路徑: {config.base_path}")
        return

    # 找出所有可用的資料集
    available_datasets = find_available_datasets(config.base_path)
    print(f"📊 發現 {len(available_datasets)} 個資料集\n")

    # 創建輸出目錄
    output_dir = os.path.join(config.output_dir, "slice_by_slice")
    screenshot_dir = os.path.join(output_dir, "screenshots")
    detailed_slices_dir = os.path.join(output_dir, "detailed_slices")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)
    os.makedirs(detailed_slices_dir, exist_ok=True)

    # 載入已知水腦症案例
    known_hydrocephalus = load_hydrocephalus_reference()
    if known_hydrocephalus:
        print(f"📋 載入 {len(known_hydrocephalus)} 個已知水腦症案例\n")

    all_results = {}
    slice_analysis_summaries = []
    success_count = 0
    fail_count = 0

    # 逐案例分析（一次處理完成單切片和詳細分析）
    for i, dataset_name in enumerate(available_datasets, 1):
        print(f"\n[{i}/{len(available_datasets)}] 處理: {dataset_name}")
        print("-" * 80)

        # 檢查檔案路徑
        paths, path_success = check_prelabeled_data_paths(config.base_path, dataset_name)
        if not path_success:
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
                fail_count += 1
                continue

        # 準備腦部遮罩
        brain_mask_path = os.path.join(paths["dataset_path"], "brain_mask_from_original.nii.gz")
        if not os.path.exists(brain_mask_path):
            brain_success = create_brain_mask_from_original(paths["original"], brain_mask_path)
            if not brain_success:
                print(f"❌ 無法建立腦部遮罩，跳過 {dataset_name}")
                fail_count += 1
                continue

        # 組裝案例路徑
        case_paths = {
            'original': paths["original"],
            'ventricles': ventricle_mask_path,
            'brain_mask': brain_mask_path
        }

        # 執行逐切片詳細分析（一次處理）
        slice_summary = run_slice_by_slice_analysis_for_case(
            dataset_name,
            case_paths,
            occupancy_threshold=config.occupancy_threshold,
            output_base_dir=detailed_slices_dir
        )

        if slice_summary and slice_summary.get('data_file'):
            # 從保存的檔案中讀取詳細切片數據
            import json
            with open(slice_summary['data_file'], 'r', encoding='utf-8') as f:
                slice_data = json.load(f)

            slice_results = slice_data['slice_details']
            evans_indices = [s['evans_analysis']['evans_index'] for s in slice_results]
            median_ei = sorted(evans_indices)[len(evans_indices) // 2]

            # 找最接近中位數的切片
            best_slice = min(slice_results, key=lambda s: abs(s['evans_analysis']['evans_index'] - median_ei))

            # 整理結果（與原本格式相容）
            result = {
                "dataset": dataset_name,
                "ventricle_segment": best_slice['ventricle_segment'],
                "skull_segment": best_slice['skull_segment'],
                "evans_analysis": best_slice['evans_analysis'],
                "files_used": {
                    "original": paths["original"],
                    "ventricles": ventricle_mask_path,
                    "brain_mask": brain_mask_path,
                    "ventricle_left": paths.get("ventricle_left", "N/A"),
                    "ventricle_right": paths.get("ventricle_right", "N/A"),
                    "needs_merge": paths["needs_merge"]
                }
            }

            # 生成最佳切片的截圖
            if generate_screenshots:
                try:
                    screenshot_success = generate_evans_slice_screenshot(
                        dataset_name,
                        paths["original"],
                        ventricle_mask_path,
                        brain_mask_path,
                        best_slice['ventricle_segment'],
                        best_slice['skull_segment'],
                        screenshot_dir
                    )
                    if screenshot_success:
                        result["screenshot_path"] = os.path.join(screenshot_dir, f'{dataset_name}_evans_slice.png')
                except Exception as e:
                    print(f"⚠️ {dataset_name}: 截圖生成失敗 - {e}")

            all_results[dataset_name] = result
            slice_analysis_summaries.append(slice_summary)
            success_count += 1
        else:
            fail_count += 1

    # 驗證結果
    if known_hydrocephalus:
        validation = validate_results_against_reference(all_results, known_hydrocephalus)
        all_results['_validation'] = validation

    # 生成逐切片分析的詳細摘要報告
    if any(s is not None for s in slice_analysis_summaries):
        detailed_report_path = os.path.join(detailed_slices_dir, "detailed_summary.md")
        generate_detailed_summary_report(slice_analysis_summaries, detailed_report_path)
        print(f"✅ 逐切片分析摘要報告已保存: {detailed_report_path}")

    # 將逐切片分析結果加入主結果中
    all_results['_slice_analysis_summaries'] = slice_analysis_summaries

    # 儲存結果
    print("\n" + "=" * 80)
    print("儲存分析結果...")
    print("=" * 80)

    results_file = os.path.join(output_dir, "slice_by_slice_analysis_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_results': all_results,
            'config': config.to_dict()
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ 結果已儲存: {results_file}")

    # 生成報告
    report_file = os.path.join(output_dir, "slice_by_slice_analysis_report.md")
    from visualization import generate_markdown_report
    validation_results = all_results.get('_validation')
    generate_markdown_report(all_results, report_file, validation_results)
    print(f"✅ 報告已生成: {report_file}")

    # 最終統計
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"✅ 成功: {success_count} 個案例")
    print(f"❌ 失敗: {fail_count} 個案例")
    print(f"📄 結果檔案: {results_file}")
    print(f"📄 報告檔案: {report_file}")
    if generate_screenshots:
        print(f"📁 截圖目錄: {screenshot_dir}")
    if detailed_slices_dir:
        print(f"📁 詳細切片分析: {detailed_slices_dir}")
    print(f"📁 輸出目錄: {output_dir}")
    print("=" * 80)


def main():
    """主程式入口"""
    args = parse_arguments()

    # 載入設定
    config = load_config(args.config)

    # 命令列參數覆蓋設定檔
    if args.base_path:
        config.base_path = args.base_path
    if args.threshold is not None:
        config.occupancy_threshold = args.threshold
    if args.output_dir:
        config.output_dir = args.output_dir

    # 列出資料集並結束
    if args.list_datasets:
        if not os.path.exists(config.base_path):
            print(f"❌ 找不到資料路徑: {config.base_path}")
            sys.exit(1)

        datasets = find_available_datasets(config.base_path)
        print(f"\n發現 {len(datasets)} 個資料集:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset}")
        sys.exit(0)

    # 儲存設定並結束
    if args.save_config:
        config.save_to_file(args.save_config)
        sys.exit(0)

    # 執行分析
    generate_screenshots = not args.no_screenshots

    try:
        if args.mode == 'global_max':
            execute_global_max_analysis(config, generate_screenshots)
        elif args.mode == 'slice_by_slice':
            execute_slice_by_slice_analysis(config, generate_screenshots)
    except KeyboardInterrupt:
        print("\n\n❌ 使用者中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
