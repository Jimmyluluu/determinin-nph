#!/usr/bin/env python3
"""
簡單的 Evans Index 分析腳本
"""
import os
from src.pipeline_for_dcm_folder import run_pipeline_dcm_to_data_folder
from src.ei_estimation import ei_estimation_pipeline

def run_evans_analysis(dcm_folder_path):
    """
    執行完整的 Evans Index 分析流程
    """
    print(f"開始分析: {dcm_folder_path}")

    # 步驟1: 執行完整的處理管線 (不需要 token)
    try:
        results = run_pipeline_dcm_to_data_folder(
            dcm_folder_path=dcm_folder_path,
            totalseg_token=None,  # 不需要 token
            project_root="."
        )

        print(f"✅ 處理完成，數據保存在: {results['output_dir']}")

        # 步驟2: 執行 Evans Index 計算
        ei_results = ei_estimation_pipeline(
            base=results['output_dir'],
            verbosity=True,
            show_visualization=True
        )

        print(f"🧠 Evans Index: {ei_results['medical_interpretation']['evans_index']}")
        print(f"📊 臨床意義: {ei_results['medical_interpretation']['clinical_significance']}")

        return ei_results

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return None

if __name__ == "__main__":
    # 選擇第一個案例進行測試
    test_case = "/Volumes/Kuro醬の1TSSD/沒有顯影劑/000016209E"

    if os.path.exists(test_case):
        results = run_evans_analysis(test_case)
    else:
        print(f"❌ 找不到路徑: {test_case}")