# 🧠 MindScope

Automatic tool for detecting **Evans Index** and analyzing brain structures from CT scans.

---

## 🚀 Usage

### 方式一：Evans Index 分析（已標記資料）

MindScope 提供兩種 Evans Index 分析模式：

#### 1. 全域最大值分析
在前角範圍內找出腦室和顱骨的全域最大寬度（可來自不同切片）：

```bash
python main.py --mode global_max
```

#### 2. 逐切片分析
在前角範圍內為每個切片計算 Evans Index：

```bash
python main.py --mode slice_by_slice
```

#### 進階選項

```bash
# 使用自訂設定檔
python main.py --mode global_max --config my_config.json

# 覆蓋特定參數
python main.py --mode global_max --base-path /path/to/data --threshold 0.7

# 不生成可視化截圖
python main.py --mode slice_by_slice --no-screenshots

# 列出所有可用的資料集
python main.py --list-datasets

# 儲存目前設定到檔案
python main.py --save-config config.json
```

#### 設定檔範例

複製 `config.example.json` 並修改參數：

```json
{
  "base_path": "/path/to/your/data",
  "occupancy_threshold": 0.6,
  "max_reasonable_width": 200,
  "output_dir": "result",
  "screenshot_output_dir": "evans_slices"
}
```

---

### 🗂️ Folder Structure

```
project-root/
├── main.py                      # 統一入口點
├── config.example.json          # 設定檔範例
├── src/
│   ├── config.py               # 設定管理
│   ├── core/                   # 核心功能
│   │   ├── measurement.py      # 共用測量邏輯
│   │   ├── validation.py       # Evans Index 計算與驗證
│   │   └── logger.py           # 日誌管理
│   ├── data_io/                # 檔案處理
│   │   └── data_loader.py      # 資料載入
│   ├── utils.py                # 向後兼容層
│   ├── evans_analysis.py       # 逐切片分析
│   ├── global_max_evans_analysis.py  # 全域最大值分析
│   ├── slice_by_slice_analysis.py
│   ├── image_processing.py
│   ├── visualization.py
│   └── pipeline_for_dcm_folder.py    # DICOM 處理
├── result/                     # 分析結果輸出
│   ├── global_max/            # 全域最大值分析結果
│   └── slice_by_slice_analysis_results.json
├── evans_slices/              # 可視化截圖
└── .env                       # TotalSegmentator token
```

---

## 📊 輸出結果

### 全域最大值分析
- `result/global_max/global_max_summary.md` - 摘要報告
- `result/global_max/{case_name}/global_max_data.json` - 每個案例的詳細數據
- `result/global_max/{case_name}/screenshots/` - 可視化截圖

### 逐切片分析
- `result/slice_by_slice_analysis_results.json` - 完整分析結果
- `result/slice_by_slice_analysis_report.md` - Markdown 報告
- `evans_slices/{case_name}.png` - Evans Index 測量截圖
