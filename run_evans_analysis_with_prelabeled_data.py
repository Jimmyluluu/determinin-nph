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
    # 如果沒有合併的 ventricles，檢查是否有左右腦室檔案
    if "ventricles" not in existing_paths:
        if "ventricle_left" in existing_paths and "ventricle_right" in existing_paths:
            existing_paths["needs_merge"] = True
        else:
            print(f"❌ {dataset_name}: 缺少腦室檔案")
            return existing_paths, False
    else:
        existing_paths["needs_merge"] = False

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

def merge_left_right_ventricles(left_path: str, right_path: str, output_path: str) -> bool:
    """
    合併左右腦室遮罩為單一檔案
    """
    try:
        import SimpleITK as sitk
        import numpy as np

        # 讀取左右腦室
        left_img = sitk.ReadImage(left_path)
        right_img = sitk.ReadImage(right_path)

        # 檢查影像尺寸是否一致
        if left_img.GetSize() != right_img.GetSize():
            return False

        # 二值化
        left_binary = sitk.BinaryThreshold(left_img, lowerThreshold=0.5, upperThreshold=10000,
                                         insideValue=1, outsideValue=0)
        right_binary = sitk.BinaryThreshold(right_img, lowerThreshold=0.5, upperThreshold=10000,
                                          insideValue=1, outsideValue=0)

        # 合併（邏輯 OR）
        merged = sitk.Or(left_binary, right_binary)

        # 保存結果
        sitk.WriteImage(merged, output_path)
        return True

    except Exception as e:
        return False

def find_best_ventricle_segment(nii_path: str, occupancy_threshold: float = 0.8, max_reasonable_width: int = 200) -> Dict:
    """
    找出腦室的最佳測量段（從現有的 notebook 程式碼複製）
    """
    img = nib.load(nii_path)
    mask_data = img.get_fdata()
    binary = (mask_data > 0).astype(np.uint8)

    best = {'width': 0, 'z': None, 'y': None, 'x1': None, 'x2': None, 'occupancy': 0}
    X, Y, Z = binary.shape


    suspicious_segments = []

    for z in range(Z):
        slice_ = binary[:, :, z]
        for y in range(Y):
            col = slice_[:, y]
            xs = np.where(col > 0)[0]
            if xs.size < 2:
                continue
            x1, x2 = xs.min(), xs.max()
            width = x2 - x1
            if width <= best['width']:
                continue

            # 檢查寬度是否合理
            if width > max_reasonable_width:
                suspicious_segments.append({
                    'width': width, 'z': z, 'y': y, 'x1': x1, 'x2': x2
                })
                continue

            occupancy = col[x1:x2+1].sum() / (width + 1)
            if occupancy >= occupancy_threshold:
                best.update({'width': int(width), 'z': int(z), 'y': int(y), 'x1': int(x1), 'x2': int(x2), 'occupancy': float(occupancy)})


    return best

def find_skull_segment(skull_path: str, z_fixed: int, min_reasonable_width: int = 100) -> Dict:
    """
    找出顱骨的最寬段（從現有的 notebook 程式碼複製）
    """
    skull_img = nib.load(skull_path)
    skull_data = skull_img.get_fdata()
    skull_binary = (skull_data > 0).astype(np.uint8)

    slice_skull = skull_binary[:, :, z_fixed]
    best_segment = None

    for y in range(slice_skull.shape[1]):
        col_skull = slice_skull[:, y]
        xs = np.where(col_skull > 0)[0]

        if xs.size < 2:
            continue

        x1_skull, x2_skull = xs.min(), xs.max()
        width_skull = x2_skull - x1_skull
        occupancy_skull = col_skull[x1_skull:x2_skull+1].sum() / (width_skull + 1)

        if (best_segment is None) or (width_skull > best_segment['width']):
            # 檢查顱骨寬度是否合理
            if width_skull < min_reasonable_width:
                continue
            best_segment = {
                'width': int(width_skull),
                'x1': int(x1_skull),
                'x2': int(x2_skull),
                'occupancy': float(occupancy_skull),
                'z': int(z_fixed),
                'y': int(y)
            }

    if best_segment is None:
        raise RuntimeError(f"在 z={z_fixed} 找不到顱骨段")

    return best_segment

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

def run_prelabeled_evans_analysis(base_path: str, dataset_name: str) -> Optional[Dict]:
    """
    使用標記好的資料執行 Evans Index 分析
    """
    print(f"\n🔍 開始分析: {dataset_name}")

    # 檢查檔案路徑
    paths, success = check_prelabeled_data_paths(base_path, dataset_name)
    if not success:
        return None

    # 準備腦室遮罩
    ventricle_mask_path = None
    if paths["needs_merge"]:
        # 需要合併左右腦室
        ventricle_mask_path = os.path.join(paths["dataset_path"], "merged_ventricles.nii.gz")
        if not os.path.exists(ventricle_mask_path):
            success = merge_left_right_ventricles(
                paths["ventricle_left"],
                paths["ventricle_right"],
                ventricle_mask_path
            )
            if not success:
                print(f"❌ 無法合併腦室遮罩，跳過 {dataset_name}")
                return None
    else:
        # 使用現有的合併腦室檔案
        ventricle_mask_path = paths["ventricles"]

    # 準備腦部遮罩 - 統一使用原始影像
    print(f"🧠 使用原始影像建立腦部遮罩...")
    brain_mask_path = os.path.join(paths["dataset_path"], "brain_mask_from_original.nii.gz")
    if not os.path.exists(brain_mask_path):
        success = create_brain_mask_from_original(paths["original"], brain_mask_path)
        if not success:
            print(f"❌ 無法建立腦部遮罩，跳過 {dataset_name}")
            return None

    try:
        # 找出最佳腦室段
        print("🔍 尋找最佳腦室測量段...")
        ventricle_segment = find_best_ventricle_segment(ventricle_mask_path)

        if ventricle_segment['width'] == 0:
            print(f"❌ 在 {dataset_name} 中找不到有效的腦室段")
            return None

        # 使用相同的 z 切片找出顱骨段
        skull_segment = find_skull_segment(brain_mask_path, ventricle_segment['z'])

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


        return results

    except Exception as e:
        return None

def batch_analyze_prelabeled_data(base_path: str, datasets: Optional[List[str]] = None) -> Dict:
    """
    批次分析多個標記資料集
    """
    if datasets is None:
        datasets = find_available_datasets(base_path)


    results = {}
    successful = 0
    failed = 0

    for dataset in datasets:
        result = run_prelabeled_evans_analysis(base_path, dataset)
        if result:
            results[dataset] = result
            successful += 1
        else:
            failed += 1

    print(f"\n分析完成: 成功 {successful}, 失敗 {failed}")

    # 顯示統計摘要
    if results:
        evans_indices = [r["evans_analysis"]["evans_index"] for r in results.values()]
        avg_evans = np.mean(evans_indices)
        high_risk_count = sum(1 for r in results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "高")
        medium_risk_count = sum(1 for r in results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "中")

        print(f"\n📊 統計摘要:")
        print(f"   平均 Evans Index: {avg_evans:.4f}")
        print(f"   腦室擴大案例: {high_risk_count}/{len(results)} ({high_risk_count/len(results)*100:.1f}%)")
        print(f"   可能/早期擴大: {medium_risk_count}/{len(results)} ({medium_risk_count/len(results)*100:.1f}%)")

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
    validation = {
        "total_analyzed": len(results),
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
            if results[case]["evans_analysis"]["hydrocephalus_risk"] == "高":
                validation["hydrocephalus_correctly_identified"] += 1
                total_correct += 1
            else:
                validation["false_negatives"].append({
                    "case": case,
                    "evans_index": results[case]["evans_analysis"]["evans_index"]
                })
        else:
            validation["not_analyzed"].append(case)

    # 檢查應該是正常的案例
    for case, result in results.items():
        if case not in known_hydrocephalus:  # 應該是正常案例
            if result["evans_analysis"]["hydrocephalus_risk"] == "低":
                validation["normal_correctly_identified"] += 1
                total_correct += 1
            else:
                validation["false_positives"].append({
                    "case": case,
                    "evans_index": result["evans_analysis"]["evans_index"]
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

        if not results:
            f.write("❌ 沒有成功分析的資料集\n")
            return

        # 統計摘要
        evans_indices = [r["evans_analysis"]["evans_index"] for r in results.values()]
        avg_evans = sum(evans_indices) / len(evans_indices)
        high_risk_count = sum(1 for r in results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "高")
        medium_risk_count = sum(1 for r in results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "中")
        normal_count = sum(1 for r in results.values() if r["evans_analysis"]["hydrocephalus_risk"] == "低")

        f.write("## 📊 統計摘要\n\n")
        f.write(f"- **總共分析案例**: {len(results)} 個\n")
        f.write(f"- **平均 Evans Index**: {avg_evans:.4f}\n")
        f.write(f"- **正常範圍 (≤ 0.25)**: {normal_count}/{len(results)} ({normal_count/len(results)*100:.1f}%)\n")
        f.write(f"- **可能/早期擴大 (0.25-0.30)**: {medium_risk_count}/{len(results)} ({medium_risk_count/len(results)*100:.1f}%)\n")
        f.write(f"- **腦室擴大 (> 0.30)**: {high_risk_count}/{len(results)} ({high_risk_count/len(results)*100:.1f}%)\n\n")

        # 分類統計
        normal_cases = [k for k, v in results.items() if v["evans_analysis"]["hydrocephalus_risk"] == "低"]
        medium_cases = [k for k, v in results.items() if v["evans_analysis"]["hydrocephalus_risk"] == "中"]
        high_cases = [k for k, v in results.items() if v["evans_analysis"]["hydrocephalus_risk"] == "高"]

        f.write("## 🟢 正常範圍案例\n\n")
        if normal_cases:
            f.write("| 案例 | Evans Index | 腦室寬度 | 顱骨寬度 |\n")
            f.write("|------|-------------|----------|----------|\n")
            for case in sorted(normal_cases):
                r = results[case]
                ei = r["evans_analysis"]["evans_index"]
                vw = r["evans_analysis"]["ventricle_width"]
                sw = r["evans_analysis"]["skull_width"]
                f.write(f"| {case} | {ei:.4f} | {vw} | {sw} |\n")
        else:
            f.write("沒有正常範圍的案例\n")

        f.write("\n## 🟡 可能/早期擴大案例\n\n")
        if medium_cases:
            f.write("| 案例 | Evans Index | 腦室寬度 | 顱骨寬度 | 臨床意義 |\n")
            f.write("|------|-------------|----------|----------|----------|\n")
            for case in sorted(medium_cases):
                r = results[case]
                ei = r["evans_analysis"]["evans_index"]
                vw = r["evans_analysis"]["ventricle_width"]
                sw = r["evans_analysis"]["skull_width"]
                cs = r["evans_analysis"]["clinical_significance"]
                f.write(f"| {case} | {ei:.4f} | {vw} | {sw} | {cs} |\n")
        else:
            f.write("沒有可能/早期擴大案例\n")

        f.write("\n## 🔴 腦室擴大案例\n\n")
        if high_cases:
            f.write("| 案例 | Evans Index | 腦室寬度 | 顱骨寬度 | 臨床意義 |\n")
            f.write("|------|-------------|----------|----------|----------|\n")
            for case in sorted(high_cases):
                r = results[case]
                ei = r["evans_analysis"]["evans_index"]
                vw = r["evans_analysis"]["ventricle_width"]
                sw = r["evans_analysis"]["skull_width"]
                cs = r["evans_analysis"]["clinical_significance"]
                f.write(f"| {case} | {ei:.4f} | {vw} | {sw} | {cs} |\n")
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

        # 說明
        f.write("\n## 📖 說明\n\n")
        f.write("- **Evans Index**: 腦室寬度與顱骨寬度的比值\n")
        f.write("- **正常範圍**: ≤ 0.25\n")
        f.write("- **可能/早期腦室擴大**: 0.25-0.30\n")
        f.write("- **腦室擴大**: > 0.30\n")
        f.write("- **測量方法**: 在相同 Z 切片上測量腦室和顱骨的最大寬度\n\n")

if __name__ == "__main__":
    # 設定標記資料的路徑
    LABELED_DATA_PATH = "/Volumes/Kuro醬の1TSSD/標記好的資料"

    if not os.path.exists(LABELED_DATA_PATH):
        print(f"找不到資料路徑: {LABELED_DATA_PATH}")
        exit(1)

    # 找出所有可用的資料集
    available_datasets = find_available_datasets(LABELED_DATA_PATH)
    print(f"發現 {len(available_datasets)} 個資料集")

    # 執行批次分析
    batch_results = batch_analyze_prelabeled_data(LABELED_DATA_PATH)

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

    print(f"\n所有結果檔案已保存在 {result_dir}/ 資料夾")
