#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

def generate_evans_slice_screenshot(case_name, original_path, ventricle_path, brain_mask_path,
                                  ventricle_coords, skull_coords, output_dir):
    """
    生成 Evans Index 計算時使用的切片截圖
    """
    try:
        # 讀取影像
        original_img = nib.load(original_path)
        original_data = original_img.get_fdata()

        ventricle_img = nib.load(ventricle_path)
        ventricle_data = ventricle_img.get_fdata()

        brain_img = nib.load(brain_mask_path)
        brain_data = brain_img.get_fdata()

        # 取得測量的 Z 切片
        z_slice = ventricle_coords['z']
        y_ventricle = ventricle_coords['y']
        y_skull = skull_coords['y']

        # 取得該切片的資料
        original_slice = original_data[:, :, z_slice]
        ventricle_slice = ventricle_data[:, :, z_slice]
        brain_slice = brain_data[:, :, z_slice]

        # 建立圖形
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 顯示原始影像（灰階）
        ax.imshow(original_slice.T, cmap='gray', origin='lower', alpha=0.8)

        # 疊加腦室遮罩（紅色）
        ventricle_mask = ventricle_slice > 0
        ax.imshow(np.ma.masked_where(~ventricle_mask.T, ventricle_slice.T),
                 cmap='Reds', origin='lower', alpha=0.6)

        # 疊加腦部邊界（藍色邊框）
        brain_mask = brain_slice > 0
        ax.contour(brain_mask.T, levels=[0.5], colors='blue', linewidths=1, alpha=0.7)

        # 畫出腦室測量線
        v_x1, v_x2 = ventricle_coords['x1'], ventricle_coords['x2']
        v_width = ventricle_coords['width']
        ax.plot([v_x1, v_x2], [y_ventricle, y_ventricle], 'r-', linewidth=3, label=f'Ventricle Width: {v_width}px')
        ax.plot([v_x1, v_x1], [y_ventricle-5, y_ventricle+5], 'r-', linewidth=2)
        ax.plot([v_x2, v_x2], [y_ventricle-5, y_ventricle+5], 'r-', linewidth=2)

        # 畫出顱骨測量線
        s_x1, s_x2 = skull_coords['x1'], skull_coords['x2']
        s_width = skull_coords['width']
        ax.plot([s_x1, s_x2], [y_skull, y_skull], 'g-', linewidth=3, label=f'Skull Width: {s_width}px')
        ax.plot([s_x1, s_x1], [y_skull-5, y_skull+5], 'g-', linewidth=2)
        ax.plot([s_x2, s_x2], [y_skull-5, y_skull+5], 'g-', linewidth=2)

        # 標註座標資訊
        ax.text(10, original_slice.shape[1] - 20,
               f'Slice: Z={z_slice}\nVentricle Y: {y_ventricle}\nSkull Y: {y_skull}',
               color='white', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

        # 設定圖形屬性
        ax.set_title(f'{case_name} - Evans Index Measurement Slice\nZ={z_slice}, Evans Index={ventricle_coords["width"]/skull_coords["width"]:.4f}',
                    fontsize=14, pad=20)
        ax.set_xlabel('X axis (pixels)')
        ax.set_ylabel('Y axis (pixels)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 顯示整個腦部切片
        ax.set_xlim(0, original_slice.shape[0])
        ax.set_ylim(0, original_slice.shape[1])

        # 保存截圖
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{case_name}_evans_slice.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated screenshot: {output_path}")
        return True

    except Exception as e:
        print(f"❌ {case_name}: Screenshot generation failed - {str(e)}")
        return False

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