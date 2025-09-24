#!/usr/bin/env python3
"""
影像處理模組
"""
import os
import numpy as np
import nibabel as nib
from typing import Dict, Optional, Tuple


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