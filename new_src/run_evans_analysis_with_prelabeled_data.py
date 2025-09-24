#!/usr/bin/env python3
"""
直接使用標記好的資料進行 Evans Index 分析
"""
import os
import numpy as np
import nibabel as nib
import json
from typing import Dict, List, Optional
import glob

# 導入可視化截圖生成函數
from generate_evans_slices import generate_evans_slice_screenshot

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

def check_prelabeled_data_paths(base_path: str, dataset_name: str) -> Dict[str, str]:
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

def create_brain_mask_from_csf(csf_path: str, output_path: str, dilation_radius: int = 10) -> bool:
    """
    從 CSF 遮罩建立腦部外輪廓遮罩
    """
    try:
        import SimpleITK as sitk

        # 讀取 CSF 遮罩
        csf_img = sitk.ReadImage(csf_path)
        csf_binary = sitk.BinaryThreshold(csf_img, lowerThreshold=0.5, upperThreshold=10000,
                                         insideValue=1, outsideValue=0)

        # 膨脹操作來建立腦部外輪廓
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(dilation_radius)
        brain_mask = dilate_filter.Execute(csf_binary)

        # 保存結果
        sitk.WriteImage(brain_mask, output_path)
        print(f"✅ 從 CSF 建立腦部遮罩: {output_path}")
        return True

    except Exception as e:
        print(f"❌ 建立腦部遮罩失敗: {e}")
        return False

def create_brain_mask_from_original(original_path: str, output_path: str) -> bool:
    """
    從原始影像建立顱內空間遮罩（用於測量顱骨寬度）
    """
    try:
        import SimpleITK as sitk
        import numpy as np

        # 讀取原始影像
        original_img = sitk.ReadImage(original_path, sitk.sitkFloat32)

        # 轉換為 numpy 陣列來計算統計資訊
        img_array = sitk.GetArrayFromImage(original_img)

        # 直接使用整個腦部，不做閾值處理

        # 建立簡單的非零遮罩
        brain_mask = sitk.BinaryThreshold(original_img,
                                        lowerThreshold=1.0,  # 任何非零值
                                        upperThreshold=100000,
                                        insideValue=1,
                                        outsideValue=0)

        # 形態學操作來建立完整的顱內空間
        # 1. 先填充所有內部空洞（包括腦室等）
        fill_holes_filter = sitk.BinaryFillholeImageFilter()
        brain_mask = fill_holes_filter.Execute(brain_mask)

        # 2. 進行閉運算來連接斷裂的區域
        closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        closing_filter.SetKernelRadius(5)
        brain_mask = closing_filter.Execute(brain_mask)

        # 3. 再次填充空洞確保完整性
        brain_mask = fill_holes_filter.Execute(brain_mask)

        # 4. 輕微膨脹確保包含完整邊界
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(2)
        brain_mask = dilate_filter.Execute(brain_mask)

        # 保存結果
        sitk.WriteImage(brain_mask, output_path)
        return True

    except Exception as e:
        return False

def merge_left_right_ventricles(left_path: str, right_path: str, output_path: str, dataset_name: str = "", original_path: str = None) -> bool:
    """
    合併左右腦室遮罩為單一檔案，處理尺寸不同的情況
    """
    try:
        import SimpleITK as sitk
        import numpy as np
        import os

        # 檢查檔案是否存在
        if not os.path.exists(left_path):
            print(f"❌ {dataset_name}: 左腦室檔案不存在: {left_path}")
            return False
        if not os.path.exists(right_path):
            print(f"❌ {dataset_name}: 右腦室檔案不存在: {right_path}")
            return False

        # 檢查檔案大小
        left_size = os.path.getsize(left_path)
        right_size = os.path.getsize(right_path)
        if left_size == 0 or right_size == 0:
            print(f"❌ {dataset_name}: 檔案大小為 0 (左: {left_size}, 右: {right_size})")
            return False

        # 讀取左右腦室
        left_img = sitk.ReadImage(left_path)
        right_img = sitk.ReadImage(right_path)

        # 檢查影像尺寸是否一致
        if left_img.GetSize() != right_img.GetSize():
            print(f"🔍 {dataset_name}: 左右腦室影像尺寸不一致 (左: {left_img.GetSize()}, 右: {right_img.GetSize()})")

            # 如果有原始影像，使用它作為參考空間
            if original_path and os.path.exists(original_path):
                print(f"🔧 {dataset_name}: 使用原始影像作為參考空間進行重新採樣...")
                original_img = sitk.ReadImage(original_path)

                # 重新採樣左右腦室到原始影像空間
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(original_img)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 使用最近鄰插值保持二進制遮罩
                resampler.SetDefaultPixelValue(0)

                left_resampled = resampler.Execute(left_img)
                right_resampled = resampler.Execute(right_img)

                # 轉換為 numpy 進行合併
                left_array = sitk.GetArrayFromImage(left_resampled)
                right_array = sitk.GetArrayFromImage(right_resampled)

                # 合併
                merged_array = np.logical_or(left_array > 0.5, right_array > 0.5).astype(np.uint8)

                # 轉回 SimpleITK 影像
                merged = sitk.GetImageFromArray(merged_array)
                merged.CopyInformation(original_img)

                print(f"✅ {dataset_name}: 重新採樣後合併完成")
            else:
                print(f"❌ {dataset_name}: 無法處理尺寸不一致且缺少原始影像參考")
                return False
        else:
            # 尺寸一致的正常處理
            left_array = sitk.GetArrayFromImage(left_img)
            right_array = sitk.GetArrayFromImage(right_img)

            # 合併
            merged_array = np.logical_or(left_array > 0.5, right_array > 0.5).astype(np.uint8)

            # 轉回 SimpleITK 影像
            merged = sitk.GetImageFromArray(merged_array)
            merged.CopyInformation(left_img)

        # 檢查合併結果
        merged_nonzero = np.count_nonzero(merged_array)
        if merged_nonzero == 0:
            print(f"❌ {dataset_name}: 合併後遮罩為空")
            return False

        print(f"🔍 {dataset_name}: 合併後非零數 {merged_nonzero}")

        # 保存結果
        sitk.WriteImage(merged, output_path)
        print(f"✅ {dataset_name}: 合併完成，保存至 {output_path}")
        return True

    except Exception as e:
        print(f"❌ {dataset_name}: 合併失敗 - {str(e)}")
        return False

def find_frontal_horns_segment(nii_path: str, dataset_name: str = "", max_reasonable_width: int = 200, occupancy_threshold: float = 0.7) -> Dict:
    """
    找出側腦室前角的測量段 - 根據資料來源適應不同座標系統

    Parameters:
        nii_path (str): 腦室遮罩路徑
        dataset_name (str): 資料集名稱，用於判斷座標系統
        max_reasonable_width (int): 最大合理寬度，預設200
        occupancy_threshold (float): 佔有率閾值，預設0.7
    """
    img = nib.load(nii_path)
    mask_data = img.get_fdata()
    binary = (mask_data > 0).astype(np.uint8)

    best = {'width': 0, 'z': None, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}
    X, Y, Z = binary.shape

    # 檢查遮罩是否有內容
    total_pixels = np.count_nonzero(binary)
    if total_pixels == 0:
        print(f"❌ 腦室遮罩完全為空")
        return best

    # 找出腦室的 Z 軸範圍
    z_coords = []
    for z in range(Z):
        if np.count_nonzero(binary[:, :, z]) > 0:
            z_coords.append(z)

    if not z_coords:
        return best

    z_min, z_max = min(z_coords), max(z_coords)
    z_range = z_max - z_min

    # 前角區域：Z 軸前部 (z/3 到 z)
    if z_range > 0:
        target_z_start = int(z_min + z_range / 3)  # z/3
        target_z_end = z_max  # z
    else:
        target_z_start = target_z_end = z_min

    # 根據資料來源決定 Y 軸搜索範圍
    y_mid = Y // 2  # Y軸中點

    # data 開頭的檔案：前角在下半部 (0 到 n/2)
    # 其他檔案：前角在上半部 (n/2 到 n)
    is_data_series = dataset_name.startswith('data_')

    if is_data_series:
        y_search_range = range(0, y_mid)
        search_description = "下半部 (data 系列)"
    else:
        y_search_range = range(y_mid, Y)
        search_description = "上半部 (編號系列)"

    for z in range(target_z_start, min(target_z_end + 1, Z)):
        slice_ = binary[:, :, z]

        if np.count_nonzero(slice_) == 0:
            continue

        # 根據資料類型搜索對應的 Y 範圍
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

            # 只有佔有率達到閾值且寬度更寬時才更新
            if occupancy >= occupancy_threshold and width > best['width']:
                best.update({
                    'width': int(width),
                    'z': int(z),
                    'y': int(y),
                    'x1': int(x1),
                    'x2': int(x2),
                    'occupancy': float(occupancy)
                })

    # 調試資訊
    if best['width'] == 0:
        print(f"❌ 在前角區域找不到有效段，Z範圍: {target_z_start}-{target_z_end}, Y範圍: {search_description}")
        # 回退到全域搜尋最寬段
        return find_widest_segment_fallback(binary, max_reasonable_width, occupancy_threshold)
    else:
        print(f"✅ 找到前角段: 寬度={best['width']}, Z={best['z']}, Y={best['y']} ({search_description})")

    return best

def find_widest_segment_fallback(binary, max_reasonable_width, occupancy_threshold=0.7):
    """回退方法：找最寬的腦室段"""
    best = {'width': 0, 'z': None, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}
    X, Y, Z = binary.shape

    for z in range(Z):
        slice_ = binary[:, :, z]
        for y in range(Y):
            col = slice_[:, y]
            xs = np.where(col > 0)[0]
            if xs.size < 2:
                continue
            x1, x2 = xs.min(), xs.max()
            width = x2 - x1

            if width <= best['width'] or width > max_reasonable_width:
                continue

            occupancy = col[x1:x2+1].sum() / (width + 1)

            # 只有佔有率達到閾值才更新
            if occupancy >= occupancy_threshold:
                best.update({'width': int(width), 'z': int(z), 'y': int(y), 'x1': int(x1), 'x2': int(x2), 'occupancy': float(occupancy)})

    return best

def find_skull_segment(skull_path: str, z_fixed: int, y_ventricle: int, min_reasonable_width: int = 100) -> Dict:
    """
    在指定的 Z 切片上找到顱骨的最大寬度，而不是固定在腦室的 Y 座標
    """
    skull_img = nib.load(skull_path)
    skull_data = skull_img.get_fdata()
    skull_binary = (skull_data > 0).astype(np.uint8)

    slice_skull = skull_binary[:, :, z_fixed]

    # 在整個 Z 切片上找到顱骨的最大寬度
    max_width = 0
    best_y = y_ventricle
    best_x1, best_x2 = 0, 0
    best_occupancy = 0.0

    for y in range(slice_skull.shape[1]):
        col_skull = slice_skull[:, y]
        xs = np.where(col_skull > 0)[0]

        if xs.size >= 2:
            x1, x2 = xs.min(), xs.max()
            width = x2 - x1
            occupancy = col_skull[x1:x2+1].sum() / (width + 1)

            # 更新最大寬度
            if width > max_width:
                max_width = width
                best_y = y
                best_x1, best_x2 = x1, x2
                best_occupancy = occupancy

    # 檢查是否找到合理的顱骨寬度
    if max_width < min_reasonable_width:
        raise RuntimeError(f"在 z={z_fixed} 找到的最大顱骨寬度 {max_width} 小於最小合理值 {min_reasonable_width}")

    return {
        'width': int(max_width),
        'x1': int(best_x1),
        'x2': int(best_x2),
        'occupancy': float(best_occupancy),
        'z': int(z_fixed),
        'y': int(best_y)
    }

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
            evans_indices = [r["evans_analysis"]["evans_index"] for r in actual_results.values()]
            avg_evans = np.mean(evans_indices)
            high_risk_count = sum(1 for r in actual_results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "高")
            medium_risk_count = sum(1 for r in actual_results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "中")

            print(f"\n📊 統計摘要:")
            print(f"   平均 Evans Index: {avg_evans:.4f}")
            print(f"   腦室擴大案例: {high_risk_count}/{len(actual_results)} ({high_risk_count/len(actual_results)*100:.1f}%)")
            print(f"   可能/早期擴大: {medium_risk_count}/{len(actual_results)} ({medium_risk_count/len(actual_results)*100:.1f}%)")

    return results

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
    except Exception as e:
        return []

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

def generate_markdown_report(results: Dict, output_file: str, validation: Dict = None):
    """
    產生簡潔明瞭的 Markdown 報告
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Evans Index 分析報告\n\n")
        f.write(f"📅 分析時間: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 分離失敗案例和成功結果
        failed_cases = results.get("_failed_cases", [])
        actual_results = {k: v for k, v in results.items() if not k.startswith('_')}

        if not actual_results:
            f.write("❌ 沒有成功分析的資料集\n")
            if failed_cases:
                f.write(f"\n### ❌ 分析失敗案例 ({len(failed_cases)} 個)\n\n")
                for case in failed_cases:
                    f.write(f"- {case}\n")
            return

        # 統計摘要
        evans_indices = [r["evans_analysis"]["evans_index"] for r in actual_results.values()]
        avg_evans = sum(evans_indices) / len(evans_indices)
        high_risk_count = sum(1 for r in actual_results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "高")
        medium_risk_count = sum(1 for r in actual_results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "中")
        normal_count = sum(1 for r in actual_results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "低")

        f.write("## 📊 統計摘要\n\n")
        f.write(f"- **總共分析案例**: {len(actual_results)} 個\n")
        if failed_cases:
            f.write(f"- **分析失敗案例**: {len(failed_cases)} 個\n")
        f.write(f"- **平均 Evans Index**: {avg_evans:.4f}\n")
        f.write(f"- **正常範圍 (≤ 0.25)**: {normal_count}/{len(actual_results)} ({normal_count/len(actual_results)*100:.1f}%)\n")
        f.write(f"- **可能/早期擴大 (0.25-0.30)**: {medium_risk_count}/{len(actual_results)} ({medium_risk_count/len(actual_results)*100:.1f}%)\n")
        f.write(f"- **腦室擴大 (> 0.30)**: {high_risk_count}/{len(actual_results)} ({high_risk_count/len(actual_results)*100:.1f}%)\n\n")

        # 分類統計
        normal_cases = [k for k, v in actual_results.items() if v["evans_analysis"]["hydrocephalus_risk"] == "低"]
        medium_cases = [k for k, v in actual_results.items() if v["evans_analysis"]["hydrocephalus_risk"] == "中"]
        high_cases = [k for k, v in actual_results.items() if v["evans_analysis"]["hydrocephalus_risk"] == "高"]

        f.write("## 🟢 正常範圍案例\n\n")
        if normal_cases:
            f.write("| 案例 | Evans Index | 腦室寬度 | 顱骨寬度 | 測量位置 (x,y,z) |\n")
            f.write("|------|-------------|----------|----------|------------------|\n")
            for case in sorted(normal_cases):
                r = actual_results[case]
                ei = r["evans_analysis"]["evans_index"]
                vw = r["evans_analysis"]["ventricle_width"]
                sw = r["evans_analysis"]["skull_width"]
                vs = r["ventricle_segment"]
                ss = r["skull_segment"]
                v_pos = f"({vs['x1']}-{vs['x2']},{vs['y']},{vs['z']})"
                s_pos = f"({ss['x1']}-{ss['x2']},{ss['y']},{ss['z']})"
                f.write(f"| {case} | {ei:.4f} | {vw} | {sw} | V:{v_pos} S:{s_pos} |\n")
        else:
            f.write("沒有正常範圍的案例\n")

        f.write("\n## 🟡 可能/早期擴大案例\n\n")
        if medium_cases:
            f.write("| 案例 | Evans Index | 腦室寬度 | 顱骨寬度 | 測量位置 (x,y,z) | 臨床意義 |\n")
            f.write("|------|-------------|----------|----------|------------------|----------|\n")
            for case in sorted(medium_cases):
                r = actual_results[case]
                ei = r["evans_analysis"]["evans_index"]
                vw = r["evans_analysis"]["ventricle_width"]
                sw = r["evans_analysis"]["skull_width"]
                cs = r["evans_analysis"]["clinical_significance"]
                vs = r["ventricle_segment"]
                ss = r["skull_segment"]
                v_pos = f"({vs['x1']}-{vs['x2']},{vs['y']},{vs['z']})"
                s_pos = f"({ss['x1']}-{ss['x2']},{ss['y']},{ss['z']})"
                f.write(f"| {case} | {ei:.4f} | {vw} | {sw} | V:{v_pos} S:{s_pos} | {cs} |\n")
        else:
            f.write("沒有可能/早期擴大案例\n")

        f.write("\n## 🔴 腦室擴大案例\n\n")
        if high_cases:
            f.write("| 案例 | Evans Index | 腦室寬度 | 顱骨寬度 | 測量位置 (x,y,z) | 臨床意義 |\n")
            f.write("|------|-------------|----------|----------|------------------|----------|\n")
            for case in sorted(high_cases):
                r = actual_results[case]
                ei = r["evans_analysis"]["evans_index"]
                vw = r["evans_analysis"]["ventricle_width"]
                sw = r["evans_analysis"]["skull_width"]
                cs = r["evans_analysis"]["clinical_significance"]
                vs = r["ventricle_segment"]
                ss = r["skull_segment"]
                v_pos = f"({vs['x1']}-{vs['x2']},{vs['y']},{vs['z']})"
                s_pos = f"({ss['x1']}-{ss['x2']},{ss['y']},{ss['z']})"
                f.write(f"| {case} | {ei:.4f} | {vw} | {sw} | V:{v_pos} S:{s_pos} | {cs} |\n")
        else:
            f.write("沒有腦室擴大案例\n")

        # 驗證結果（如果有的話）
        if validation:
            f.write("\n## 🔍 驗證結果\n\n")
            f.write(f"- **總體準確率**: {validation['accuracy']:.1%}\n")
            f.write(f"- **水腦症正確識別**: {validation['hydrocephalus_correctly_identified']}/{validation['known_hydrocephalus_count']}\n")
            f.write(f"- **正常案例正確識別**: {validation['normal_correctly_identified']}\n")

            if validation['false_negatives']:
                f.write(f"- **漏報案例**: {len(validation['false_negatives'])} 個\n")

            if validation['false_positives']:
                f.write(f"- **誤報案例**: {len(validation['false_positives'])} 個\n")

            if validation['not_analyzed']:
                f.write(f"- **未分析案例**: {len(validation['not_analyzed'])} 個\n")

            # 詳細顯示已知水腦症案例的預測結果
            f.write("\n### 📋 已知水腦症案例預測狀況\n\n")
            f.write("| 案例 | Evans Index | 預測結果 | 實際狀況 | 狀態 |\n")
            f.write("|------|-------------|----------|----------|------|\n")

            # 使用驗證結果中的已知案例清單
            known_cases = [
                "000235496D", "000206288G", "000152785B",
                "000137208D", "000096384I", "000087554H"
            ]

            for case in known_cases:
                if case in results:
                    r = actual_results[case]
                    evans_index = r["evans_analysis"]["evans_index"]
                    predicted = r["evans_analysis"]["hydrocephalus_risk"]
                    status = "✅ 正確" if evans_index > 0.30 else "❌ 漏報"
                    f.write(f"| {case} | {evans_index:.4f} | {predicted} 風險 | 有水腦症 | {status} |\n")
                else:
                    f.write(f"| {case} | - | 未分析 | 有水腦症 | ❌ 未分析 |\n")

        # 說明
        f.write("\n## 📖 說明\n\n")
        f.write("- **Evans Index**: 腦室寬度與顱骨寬度的比值\n")
        f.write("- **正常範圍**: ≤ 0.25\n")
        f.write("- **可能/早期腦室擴大**: 0.25-0.30\n")
        f.write("- **腦室擴大**: > 0.30\n")
        f.write("- **測量方法**: 在相同 Z 切片上測量腦室和顱骨的最大寬度\n\n")

        # 失敗案例清單（如果有的話）
        if failed_cases:
            f.write("## ❌ 分析失敗案例\n\n")
            f.write(f"以下 {len(failed_cases)} 個案例分析失敗:\n\n")
            for case in sorted(failed_cases):
                f.write(f"- {case}\n")
            f.write("\n**失敗原因可能包括**: 缺少必要檔案、腦室遮罩問題、或測量參數超出合理範圍\n\n")

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
