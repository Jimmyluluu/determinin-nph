# 🧠 MindScope 專案說明文件

## 專案概述

MindScope 是一個基於 Python 3.10 的自動化腦部 CT 掃描分析工具，專門用於檢測 **Evans Index (伊凡斯指數)** 和分析腦部結構，以協助診斷水腦症等腦部疾病。

### 核心功能
- 🧩 自動腦部結構分割
- 📐 醫學影像對齊校正
- 📊 Evans Index 和 Ventricle-to-Brain Ratio (VBR) 計算
- 🖼️ 3D 視覺化分析結果

## 技術架構

### 系統架構
```
MindScope/
├── src/                        # 核心程式碼
│   ├── pipeline_for_dcm_folder.py    # 主要處理流程
│   └── segmentator/            # 分割與處理模組
│       ├── convert_dicom.py   # DICOM 轉換
│       ├── run_brain_segmentation.py # 腦部分割
│       ├── run_aligner_flirt.py      # 影像對齊
│       └── utils.py            # 工具函數
├── notebooks/                  # 分析與視覺化
│   └── demo_evans_index.ipynb # Evans Index 計算演示
├── data/                       # 資料目錄
│   └── brain_blueprint/        # MNI 標準腦部模板
├── tests/                      # 測試程式
└── images/                     # 範例圖片
```

### 技術堆疊

#### 核心技術
- **Python 3.10** - 主要開發語言
- **TotalSegmentator 2.9.0** - 先進的醫學影像分割工具
- **FSL FLIRT** - 醫學影像對齊工具
- **dcm2niix** - DICOM 到 NIfTI 格式轉換工具

#### 主要 Python 套件
- **SimpleITK** - 醫學影像處理
- **nibabel 5.3.2** - NIfTI 格式讀寫
- **numpy 1.26.4** - 數值運算
- **plotly 6.2.0** - 互動式視覺化
- **torch 2.7.1** - 深度學習框架（用於 TotalSegmentator）
- **dicom2nifti 2.6.1** - DICOM 格式處理
- **python-dotenv** - 環境變數管理

## 處理流程

### 主要管線流程 (pipeline_for_dcm_folder.py)

1. **DICOM 轉換**
   - 輸入：DICOM 格式的 CT 掃描資料夾
   - 使用 `dcm2niix` 將 DICOM 檔案轉換為 NIfTI 格式
   - 輸出：`head_ct.nii.gz`

2. **腦部結構分割**
   - 使用 TotalSegmentator 進行兩種分割任務：
     - `brain_structures`：分割整體腦部結構（額葉、頂葉、枕葉、顳葉、小腦等）
     - `ventricle_parts`：分割腦室系統
   - 輸出：各結構的分割遮罩檔案

3. **結構合併**
   - 合併腦部結構遮罩 → `brain_mask_combined.nii.gz`
   - 合併側腦室遮罩 → `lateral_ventricles_combined.nii.gz`

4. **影像對齊**
   - 使用 FSL FLIRT 將腦部遮罩對齊至 MNI152 標準模板
   - 計算旋轉矩陣並儲存為 `aligning.mat`
   - 修正頭部傾斜角度（pitch correction）

5. **統計分析**
   - 計算各腦部結構體積
   - 輸出 `statistics.json` 包含所有測量數據

### Evans Index 計算流程 (demo_evans_index.ipynb)

1. **讀取對齊參數**
   - 從 `aligning.mat` 提取歐拉角（roll, pitch, yaw）
   - 應用旋轉校正到腦部和腦室遮罩

2. **尋找最佳測量位置**
   - 自動定位腦室最寬的橫切面
   - 確保測量位置的準確性（佔用率閾值 > 0.9）

3. **計算 Evans Index**
   - 測量側腦室前角最大寬度
   - 測量同一層面的顱內最大寬度
   - 計算比值：Evans Index = 腦室寬度 / 顱內寬度
   - 正常值 ≤ 0.3，> 0.3 可能提示水腦症

4. **計算 VBR (Ventricle-to-Brain Ratio)**
   - VBR = 腦室體積 / 腦實質體積
   - 正常值 < 0.03，> 0.03 可能提示異常

5. **視覺化結果**
   - 3D 腦部遮罩對比（對齊前後）
   - 2D 切面測量線顯示
   - 互動式 Plotly 圖表

## 關鍵演算法

### 影像對齊演算法
```python
# 使用 FSL FLIRT 進行剛體對齊
# 參考：MNI152_T1_1mm_brain.nii 標準模板
# 輸出：4x4 變換矩陣
```

### Evans Index 自動測量
```python
# 1. 自動搜尋最佳測量層面
# 2. 確保測量線穿過腦室最寬處
# 3. 計算腦室寬度與顱內寬度比值
```

## 使用方式

### 環境設定
1. 安裝 Python 3.10 及相依套件
2. 安裝外部工具：TotalSegmentator、FSL、dcm2niix
3. 設定 `.env` 檔案含 TotalSegmentator token

### 執行分析
```bash
# 處理 DICOM 資料夾
python src/pipeline_for_dcm_folder.py --input /path/to/dicom/folder

# 開啟 Jupyter notebook 進行視覺化分析
jupyter notebook notebooks/demo_evans_index.ipynb
```

## 輸出檔案說明

| 檔案名稱 | 說明 |
|---------|------|
| `head_ct.nii.gz` | 轉換後的 CT 影像 |
| `brain_mask_combined.nii.gz` | 合併的腦部遮罩 |
| `lateral_ventricles_combined.nii.gz` | 合併的側腦室遮罩 |
| `brain_mask_aligned.nii.gz` | 對齊後的腦部遮罩 |
| `lateral_ventricles_aligned.nii.gz` | 對齊後的腦室遮罩 |
| `aligning.mat` | FSL FLIRT 對齊矩陣 |
| `statistics.json` | 體積統計與計算指標 |
| `file_paths.json` | 所有檔案路徑記錄 |

## 臨床意義

### Evans Index
- **正常範圍**：≤ 0.3
- **異常範圍**：> 0.3（可能提示水腦症）
- **測量位置**：側腦室前角最寬處

### VBR (Ventricle-to-Brain Ratio)
- **正常範圍**：< 0.03
- **異常範圍**：> 0.03（可能提示腦萎縮或水腦症）

## 系統需求

### 硬體需求
- CPU：建議多核心處理器
- RAM：至少 8GB（建議 16GB）
- 儲存空間：每個案例約需 500MB

### 軟體需求
- 作業系統：Linux、macOS、Windows（WSL）
- Python 3.10+
- CUDA（可選，用於 GPU 加速）

## 專案特色

1. **全自動化處理**：從 DICOM 輸入到分析結果完全自動化
2. **精確對齊**：使用 FSL FLIRT 確保測量準確性
3. **雙重驗證**：同時計算 Evans Index 和 VBR 提高診斷可靠性
4. **視覺化分析**：提供互動式 3D/2D 視覺化工具
5. **標準化流程**：基於 MNI152 標準模板確保結果一致性

## 限制與注意事項

1. **資料品質**：需要高品質的頭部 CT 掃描
2. **Token 需求**：TotalSegmentator 需要有效的授權 token
3. **計算資源**：深度學習模型需要較多計算資源
4. **臨床使用**：結果僅供參考，最終診斷需由專業醫師判定

## 未來發展方向

- 加入更多腦部病變檢測功能
- 支援 MRI 影像分析
- 開發圖形化使用者介面
- 整合雲端處理能力
- 增加批次處理功能

## 聯絡資訊

專案 GitHub：https://github.com/maratNeSlaiv/MindScope

---

*本文件生成日期：2025-08-12*