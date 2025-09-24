#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import os

def analyze_image_orientation(image_path, mask_path=None):
    """分析影像的軸向和座標系統"""

    # 讀取影像
    img = nib.load(image_path)
    data = img.get_fdata()

    print(f"=== 分析: {os.path.basename(image_path)} ===")
    print(f"影像形狀: {data.shape}")
    print(f"軸順序: X={data.shape[0]}, Y={data.shape[1]}, Z={data.shape[2]}")

    # 檢查仿射矩陣
    affine = img.affine
    print(f"仿射矩陣:")
    print(affine)

    # 檢查像素間距
    pixdim = img.header.get_zooms()
    print(f"像素間距: {pixdim}")

    # 分析每個軸的特徵
    print(f"\n各軸特徵分析:")
    for i, axis_name in enumerate(['X軸', 'Y軸', 'Z軸']):
        axis_size = data.shape[i]
        print(f"{axis_name}: 大小={axis_size}, 間距={pixdim[i]:.3f}mm")

    # 如果有遮罩，分析遮罩分佈
    if mask_path and os.path.exists(mask_path):
        print(f"\n=== 遮罩分析: {os.path.basename(mask_path)} ===")
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()

        # 找出遮罩的邊界
        coords = np.where(mask_data > 0)
        if len(coords[0]) > 0:
            for i, axis_name in enumerate(['X軸', 'Y軸', 'Z軸']):
                min_coord = coords[i].min()
                max_coord = coords[i].max()
                span = max_coord - min_coord
                print(f"{axis_name}: 範圍 {min_coord}-{max_coord}, 跨度={span}")

        # 在每個軸上測量最大寬度
        print(f"\n各軸方向最大寬度:")
        for axis in range(3):
            max_width = 0
            for slice_idx in range(mask_data.shape[axis]):
                if axis == 0:
                    slice_data = mask_data[slice_idx, :, :]
                elif axis == 1:
                    slice_data = mask_data[:, slice_idx, :]
                else:
                    slice_data = mask_data[:, :, slice_idx]

                # 在這個切片中找最大連續寬度
                for row in range(slice_data.shape[0]):
                    if axis == 0:
                        line = slice_data[row, :]
                    else:
                        line = slice_data[row, :] if axis == 1 else slice_data[:, row]

                    # 找連續的非零區間
                    nonzero = np.where(line > 0)[0]
                    if len(nonzero) > 0:
                        width = nonzero.max() - nonzero.min()
                        max_width = max(max_width, width)

            axis_names = ['X軸(左右)', 'Y軸(前後)', 'Z軸(上下)']
            print(f"{axis_names[axis]}: 最大寬度 = {max_width}")

if __name__ == "__main__":
    # 分析多個案例
    cases = ["001765014F", "002001468A", "data_11", "000016209E"]
    base_path = "/Volumes/Kuro醬の1TSSD/標記好的資料"

    for case in cases:
        print(f"\n{'='*50}")

        if case.startswith('data_'):
            case_num = case.split('_')[1]
            original_path = f"{base_path}/{case}/original_{case_num}.nii.gz"
        else:
            original_path = f"{base_path}/{case}/original.nii.gz"

        ventricle_path = f"{base_path}/{case}/merged_lateral_ventricles.nii.gz"

        if os.path.exists(original_path):
            analyze_image_orientation(original_path, ventricle_path)
        else:
            print(f"找不到檔案: {original_path}")