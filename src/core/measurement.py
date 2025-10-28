#!/usr/bin/env python3
"""
共用測量邏輯模組
抽取 global_max 和 slice_by_slice 的重複程式碼
"""
import numpy as np
import nibabel as nib
from typing import Tuple, Dict, Optional


def calculate_frontal_horn_range(binary_mask: np.ndarray, frontal_horn_ratio: float = 1/3) -> Tuple[int, int]:
    """
    計算前角區域的 Z 軸範圍

    Parameters:
        binary_mask (np.ndarray): 二值化遮罩
        frontal_horn_ratio (float): 前角區域起始比例，預設 1/3

    Returns:
        Tuple[int, int]: (z_start, z_end) 前角區域的 Z 軸範圍
    """
    Z = binary_mask.shape[2]

    # 找出腦室的 Z 軸範圍
    z_coords = []
    for z in range(Z):
        if np.count_nonzero(binary_mask[:, :, z]) > 0:
            z_coords.append(z)

    if not z_coords:
        return 0, 0

    z_min, z_max = min(z_coords), max(z_coords)
    z_range = z_max - z_min

    # 前角區域：Z 軸前部
    if z_range > 0:
        target_z_start = int(z_min + z_range * frontal_horn_ratio)
        target_z_end = z_max
    else:
        target_z_start = target_z_end = z_min

    return target_z_start, target_z_end


def determine_y_search_range(dataset_name: str, y_dimension: int) -> Tuple[range, str]:
    """
    根據資料集名稱決定 Y 軸搜索範圍

    Parameters:
        dataset_name (str): 資料集名稱
        y_dimension (int): Y 軸維度大小

    Returns:
        Tuple[range, str]: (搜索範圍, 描述文字)
    """
    y_mid = y_dimension // 2
    is_data_series = dataset_name.startswith('data_')

    if is_data_series:
        y_search_range = range(0, y_mid)
        search_description = "下半部 (data 系列)"
    else:
        y_search_range = range(y_mid, y_dimension)
        search_description = "上半部 (編號系列)"

    return y_search_range, search_description


def validate_measurement(width: int, occupancy: float,
                        min_width: int = 5,
                        max_width: int = 200,
                        occupancy_threshold: float = 0.6) -> bool:
    """
    驗證測量是否有效

    Parameters:
        width (int): 測量寬度
        occupancy (float): 佔有率
        min_width (int): 最小寬度閾值
        max_width (int): 最大寬度閾值
        occupancy_threshold (float): 佔有率閾值

    Returns:
        bool: 測量是否有效
    """
    # 檢查寬度合理性
    if width < min_width or width > max_width:
        return False

    # 檢查佔有率
    if occupancy < occupancy_threshold:
        return False

    return True


def load_binary_mask(nii_path: str, dataset_name: str = "") -> Optional[np.ndarray]:
    """
    載入並二值化 NIfTI 遮罩

    Parameters:
        nii_path (str): NIfTI 檔案路徑
        dataset_name (str): 資料集名稱（用於錯誤訊息）

    Returns:
        Optional[np.ndarray]: 二值化遮罩，如果失敗則回傳 None
    """
    try:
        img = nib.load(nii_path)
        mask_data = img.get_fdata()
        binary = (mask_data > 0).astype(np.uint8)

        # 檢查遮罩是否有內容
        total_pixels = np.count_nonzero(binary)
        if total_pixels == 0:
            print(f"❌ {dataset_name}: 遮罩完全為空")
            return None

        return binary

    except Exception as e:
        print(f"❌ {dataset_name}: 載入遮罩失敗 - {e}")
        return None


def calculate_occupancy(column: np.ndarray, x1: int, x2: int) -> float:
    """
    計算一維陣列在指定範圍內的佔有率

    Parameters:
        column (np.ndarray): 一維陣列
        x1 (int): 起始索引
        x2 (int): 結束索引

    Returns:
        float: 佔有率 (0.0 到 1.0)
    """
    width = x2 - x1
    if width <= 0:
        return 0.0

    occupied = column[x1:x2+1].sum()
    return float(occupied / (width + 1))


def find_max_width_in_slice(slice_2d: np.ndarray,
                            y_search_range: range,
                            min_width: int = 5,
                            max_width: int = 200,
                            occupancy_threshold: float = 0.6) -> Dict:
    """
    在指定切片中找出最大有效寬度

    Parameters:
        slice_2d (np.ndarray): 2D 切片陣列
        y_search_range (range): Y 軸搜索範圍
        min_width (int): 最小寬度閾值
        max_width (int): 最大寬度閾值
        occupancy_threshold (float): 佔有率閾值

    Returns:
        Dict: 包含 width, y, x1, x2, occupancy 的測量結果
    """
    best_measurement = {
        'width': 0,
        'y': None,
        'x1': None,
        'x2': None,
        'occupancy': 0
    }

    for y in y_search_range:
        col = slice_2d[:, y]
        xs = np.where(col > 0)[0]

        if xs.size < 2:
            continue

        x1, x2 = xs.min(), xs.max()
        width = x2 - x1

        # 檢查寬度合理性
        if width < min_width or width > max_width:
            continue

        # 計算佔有率
        occupancy = calculate_occupancy(col, x1, x2)

        # 更新最佳測量
        if occupancy >= occupancy_threshold and width > best_measurement['width']:
            best_measurement.update({
                'width': int(width),
                'y': int(y),
                'x1': int(x1),
                'x2': int(x2),
                'occupancy': float(occupancy)
            })

    return best_measurement


def get_mask_shape(nii_path: str) -> Optional[Tuple[int, int, int]]:
    """
    取得 NIfTI 檔案的形狀

    Parameters:
        nii_path (str): NIfTI 檔案路徑

    Returns:
        Optional[Tuple[int, int, int]]: (X, Y, Z) 維度，如果失敗則回傳 None
    """
    try:
        img = nib.load(nii_path)
        return img.shape
    except Exception as e:
        print(f"❌ 無法讀取檔案形狀: {e}")
        return None


def print_measurement_summary(measurement: Dict, measurement_type: str = "測量") -> None:
    """
    列印測量結果摘要

    Parameters:
        measurement (Dict): 測量結果字典
        measurement_type (str): 測量類型描述
    """
    if measurement.get('width', 0) > 0:
        print(f"  ✅ {measurement_type}: "
              f"寬度={measurement['width']}, "
              f"位置=(x:{measurement['x1']}-{measurement['x2']}, "
              f"y:{measurement['y']}, "
              f"z:{measurement.get('z', 'N/A')}), "
              f"佔有率={measurement['occupancy']:.2f}")
    else:
        print(f"  ❌ {measurement_type}: 未找到有效測量")
