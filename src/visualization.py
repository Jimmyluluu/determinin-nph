#!/usr/bin/env python3
"""
視覺化模組
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict


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


def generate_global_max_screenshot(case_name, original_path, ventricle_path, brain_mask_path,
                                   ventricle_max, skull_max, evans_index, output_dir):
    """
    生成全域最大值分析的並排切片截圖
    顯示腦室最大的切片和顱骨最大的切片
    """
    try:
        # 讀取影像
        original_img = nib.load(original_path)
        original_data = original_img.get_fdata()

        ventricle_img = nib.load(ventricle_path)
        ventricle_data = ventricle_img.get_fdata()

        brain_img = nib.load(brain_mask_path)
        brain_data = brain_img.get_fdata()

        # 判斷是否在同一切片
        same_slice = ventricle_max['z'] == skull_max['z']

        if same_slice:
            # 如果在同一切片,只顯示一個圖
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            z_slice = ventricle_max['z']

            # 取得該切片的資料
            original_slice = original_data[:, :, z_slice]
            ventricle_slice = ventricle_data[:, :, z_slice]
            brain_slice = brain_data[:, :, z_slice]

            # 顯示原始影像
            ax.imshow(original_slice.T, cmap='gray', origin='lower', alpha=0.8)

            # 疊加腦室遮罩
            ventricle_mask = ventricle_slice > 0
            ax.imshow(np.ma.masked_where(~ventricle_mask.T, ventricle_slice.T),
                     cmap='Reds', origin='lower', alpha=0.6)

            # 疊加腦部邊界
            brain_mask = brain_slice > 0
            ax.contour(brain_mask.T, levels=[0.5], colors='blue', linewidths=1, alpha=0.7)

            # 畫出腦室測量線
            v_x1, v_x2 = ventricle_max['x1'], ventricle_max['x2']
            v_y = ventricle_max['y']
            v_width = ventricle_max['width']
            ax.plot([v_x1, v_x2], [v_y, v_y], 'r-', linewidth=3, label=f'Ventricle Max: {v_width}px')
            ax.plot([v_x1, v_x1], [v_y-5, v_y+5], 'r-', linewidth=2)
            ax.plot([v_x2, v_x2], [v_y-5, v_y+5], 'r-', linewidth=2)

            # 畫出顱骨測量線
            s_x1, s_x2 = skull_max['x1'], skull_max['x2']
            s_y = skull_max['y']
            s_width = skull_max['width']
            ax.plot([s_x1, s_x2], [s_y, s_y], 'g-', linewidth=3, label=f'Skull Max: {s_width}px')
            ax.plot([s_x1, s_x1], [s_y-5, s_y+5], 'g-', linewidth=2)
            ax.plot([s_x2, s_x2], [s_y-5, s_y+5], 'g-', linewidth=2)

            ax.set_title(f'{case_name} - Global Max Evans Index\nZ={z_slice} (Same Slice), Evans Index={evans_index:.4f}',
                        fontsize=14, pad=20)
            ax.set_xlabel('X axis (pixels)')
            ax.set_ylabel('Y axis (pixels)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, original_slice.shape[0])
            ax.set_ylim(0, original_slice.shape[1])

        else:
            # 如果在不同切片,並排顯示兩個圖
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

            # 左圖: 腦室最大的切片
            z_vent = ventricle_max['z']
            original_slice_vent = original_data[:, :, z_vent]
            ventricle_slice_vent = ventricle_data[:, :, z_vent]
            brain_slice_vent = brain_data[:, :, z_vent]

            ax1.imshow(original_slice_vent.T, cmap='gray', origin='lower', alpha=0.8)
            ventricle_mask = ventricle_slice_vent > 0
            ax1.imshow(np.ma.masked_where(~ventricle_mask.T, ventricle_slice_vent.T),
                      cmap='Reds', origin='lower', alpha=0.6)
            brain_mask = brain_slice_vent > 0
            ax1.contour(brain_mask.T, levels=[0.5], colors='blue', linewidths=1, alpha=0.7)

            v_x1, v_x2 = ventricle_max['x1'], ventricle_max['x2']
            v_y = ventricle_max['y']
            v_width = ventricle_max['width']
            ax1.plot([v_x1, v_x2], [v_y, v_y], 'r-', linewidth=3, label=f'Ventricle Max: {v_width}px')
            ax1.plot([v_x1, v_x1], [v_y-5, v_y+5], 'r-', linewidth=2)
            ax1.plot([v_x2, v_x2], [v_y-5, v_y+5], 'r-', linewidth=2)

            ax1.set_title(f'Ventricle Maximum at Z={z_vent}', fontsize=14, pad=20)
            ax1.set_xlabel('X axis (pixels)')
            ax1.set_ylabel('Y axis (pixels)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, original_slice_vent.shape[0])
            ax1.set_ylim(0, original_slice_vent.shape[1])

            # 右圖: 顱骨最大的切片
            z_skull = skull_max['z']
            original_slice_skull = original_data[:, :, z_skull]
            ventricle_slice_skull = ventricle_data[:, :, z_skull]
            brain_slice_skull = brain_data[:, :, z_skull]

            ax2.imshow(original_slice_skull.T, cmap='gray', origin='lower', alpha=0.8)
            ventricle_mask2 = ventricle_slice_skull > 0
            ax2.imshow(np.ma.masked_where(~ventricle_mask2.T, ventricle_slice_skull.T),
                      cmap='Reds', origin='lower', alpha=0.6)
            brain_mask2 = brain_slice_skull > 0
            ax2.contour(brain_mask2.T, levels=[0.5], colors='blue', linewidths=1, alpha=0.7)

            s_x1, s_x2 = skull_max['x1'], skull_max['x2']
            s_y = skull_max['y']
            s_width = skull_max['width']
            ax2.plot([s_x1, s_x2], [s_y, s_y], 'g-', linewidth=3, label=f'Skull Max: {s_width}px')
            ax2.plot([s_x1, s_x1], [s_y-5, s_y+5], 'g-', linewidth=2)
            ax2.plot([s_x2, s_x2], [s_y-5, s_y+5], 'g-', linewidth=2)

            ax2.set_title(f'Skull Maximum at Z={z_skull}', fontsize=14, pad=20)
            ax2.set_xlabel('X axis (pixels)')
            ax2.set_ylabel('Y axis (pixels)')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, original_slice_skull.shape[0])
            ax2.set_ylim(0, original_slice_skull.shape[1])

            # 整體標題
            fig.suptitle(f'{case_name} - Global Max Evans Index = {evans_index:.4f}\n(Ventricle at Z={z_vent}, Skull at Z={z_skull})',
                        fontsize=16, y=0.98)

        # 保存截圖
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{case_name}_global_max.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated global max screenshot: {output_path}")
        return True

    except Exception as e:
        print(f"❌ {case_name}: Global max screenshot generation failed - {str(e)}")
        return False


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
                if case in actual_results:
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