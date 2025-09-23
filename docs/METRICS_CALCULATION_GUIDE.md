# 📊 Evans Index 與 VBR 計算指南

## Evans Index 計算

### 所需資料
1. **側腦室遮罩** - 用於測量腦室寬度
2. **腦部遮罩** - 用於測量顱內寬度
3. **對齊矩陣** - 確保測量在正確的軸向平面

### 計算步驟與程式碼

#### Step 1: 生成側腦室合併遮罩
**檔案**: `src/segmentator/utils.py:86-140`
```python
def combine_lateral_ventricles(input_dir: str, output_path: Optional[str] = None) -> str:
    # 合併 10 個側腦室部分：
    ventricles_of_interest = [
        'ventricle_body_left.nii.gz',          # 左側腦室體
        'ventricle_body_right.nii.gz',         # 右側腦室體
        'ventricle_frontal_horn_left.nii.gz',  # 左側前角（Evans Index 關鍵）
        'ventricle_frontal_horn_right.nii.gz', # 右側前角（Evans Index 關鍵）
        'ventricle_trigone_left.nii.gz',       # 左側三角區
        'ventricle_trigone_right.nii.gz',      # 右側三角區
        'ventricle_occipital_horn_left.nii.gz',# 左側後角
        'ventricle_occipital_horn_right.nii.gz',# 右側後角
        'ventricle_temporal_horn_left.nii.gz', # 左側下角
        'ventricle_temporal_horn_right.nii.gz',# 右側下角
    ]
    # 輸出: lateral_ventricles_combined.nii.gz
```

#### Step 2: 生成腦部合併遮罩
**檔案**: `src/segmentator/utils.py:26-83`
```python
def combine_brain_structures(input_dir: str, output_path: Optional[str] = None) -> str:
    # 合併 13 個腦部結構作為顱內邊界：
    structures_of_interest = [
        'frontal_lobe.nii.gz',      # 額葉
        'parietal_lobe.nii.gz',     # 頂葉
        'temporal_lobe.nii.gz',     # 顳葉
        'occipital_lobe.nii.gz',    # 枕葉
        'cerebellum.nii.gz',        # 小腦
        'brainstem.nii.gz',         # 腦幹
        # ... 其他深部結構
    ]
    # 輸出: brain_mask_combined.nii.gz
```

#### Step 3: 影像對齊校正
**檔案**: `src/segmentator/run_aligner_flirt.py`
```python
def run_flirt(input_path, ref_path, output_mat):
    # 使用 FSL FLIRT 生成對齊矩陣
    # 確保測量在標準軸向平面
    # 輸出: aligning.mat
```

**檔案**: `notebooks/demo_evans_index.ipynb` - Cell 2
```python
def apply_euler_rotation(input_path, output_path, roll_deg, pitch_deg, yaw_deg):
    # 應用旋轉矩陣校正頭部傾斜
    # 輸出: brain_mask_aligned.nii.gz
    # 輸出: lateral_ventricles_aligned.nii.gz
```

#### Step 4: 自動尋找最佳測量位置
**檔案**: `notebooks/demo_evans_index.ipynb` - Cell 4
```python
def find_best_ventricle_segment(nii_path: str, occupancy_threshold=0.9):
    """
    找到側腦室前角最寬的橫切面
    
    Returns:
        dict: {
            'width': 70,    # 腦室寬度（像素）
            'z': 22,        # 切片層數
            'y': 261,       # Y 座標
            'x1': 223,      # 起始 X 座標
            'x2': 293,      # 結束 X 座標
            'occupancy': 0.93  # 連續性指標
        }
    """
```

#### Step 5: Evans Index 計算
**檔案**: `notebooks/demo_evans_index.ipynb` - Cell 4
```python
def check_hydrocephalus(ventricles_width, skull_width):
    """
    計算 Evans Index
    
    公式: EI = 側腦室前角最大寬度 / 同層顱內最大寬度
    
    Returns:
        Evans Index: 0.243
        判讀: ≤0.3 正常, >0.3 可能水腦症
    """
    evans_index = ventricles_width / skull_width
```

### Evans Index 資料流程圖
```
TotalSegmentator 分割
    ↓
ventricle_parts → combine_lateral_ventricles() → lateral_ventricles_combined.nii.gz
brain_structures → combine_brain_structures() → brain_mask_combined.nii.gz
    ↓
FSL FLIRT 對齊 → aligning.mat
    ↓
apply_euler_rotation() → 對齊後的遮罩
    ↓
find_best_ventricle_segment() → 最佳測量位置
find_skull_segment() → 同層顱內寬度
    ↓
check_hydrocephalus() → Evans Index = 0.243

```

---

## VBR (Ventricle-to-Brain Ratio) 計算

### 所需資料
1. **腦室體積** - 從 TotalSegmentator 統計檔案
2. **腦實質體積** - 各腦葉體積總和
3. **statistics.json** - TotalSegmentator 自動生成的體積資料

### 計算步驟與程式碼

#### Step 1: TotalSegmentator 生成體積統計
**檔案**: `src/segmentator/run_brain_segmentation.py:7-37`
```python
def run_totalsegmentator(ct_path: str, output_path: str, token: str, task: str):
    command = [
        "TotalSegmentator",
        "-i", ct_path,
        "-o", output_path,
        "--task", task,
        "--statistics",  # 關鍵：自動計算各結構體積
        "-l", token
    ]
    # 輸出: brain_structures/statistics.json
    # 輸出: ventricles/statistics.json
```

#### Step 2: 讀取體積資料
**檔案**: `notebooks/demo_evans_index.ipynb` - Cell 13
```python
def load_brain_data(json_path):
    """
    讀取 statistics.json 檔案
    
    資料格式:
    {
        "frontal_lobe": {"volume": 150.5},
        "parietal_lobe": {"volume": 120.3},
        "temporal_lobe": {"volume": 110.2},
        "occipital_lobe": {"volume": 90.1},
        "cerebellum": {"volume": 140.8},
        "ventricle": {"volume": 31.2}
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
```

#### Step 3: VBR 計算
**檔案**: `notebooks/demo_evans_index.ipynb` - Cell 13
```python
def calculate_brain_metrics(json_path):
    """
    計算 VBR (Ventricle-to-Brain Ratio)
    """
    data = load_brain_data(json_path)
    
    # 獲取腦室體積
    ventricle_volume = get_volume(data, "ventricle")
    
    # 計算腦實質總體積（選擇主要腦葉）
    lobar_regions = [
        "frontal_lobe",    # 額葉
        "parietal_lobe",   # 頂葉
        "occipital_lobe",  # 枕葉
        "temporal_lobe",   # 顳葉
        "cerebellum"       # 小腦（如果存在）
    ]
    brain_volume = sum(get_volume(data, region) for region in lobar_regions)
    
    # 計算 VBR
    vbr = ventricle_volume / brain_volume if brain_volume > 0 else 0
    
    return {
        "brain_volume": brain_volume,      # 例: 1200 ml
        "ventricle_volume": ventricle_volume,  # 例: 31.2 ml
        "VBR": round(vbr, 4),              # 例: 0.0261
        "note": ">0.03 may indicate hydrocephalus"
    }
```

### VBR 資料流程圖
```
TotalSegmentator 分割
    ↓
--statistics 參數 → 自動計算各結構體積
    ↓
brain_structures/statistics.json
    ├── frontal_lobe: {volume: 150.5}
    ├── parietal_lobe: {volume: 120.3}
    ├── temporal_lobe: {volume: 110.2}
    ├── occipital_lobe: {volume: 90.1}
    ├── cerebellum: {volume: 140.8}
    └── ventricle: {volume: 31.2}
    ↓
calculate_brain_metrics()
    ↓
VBR = ventricle_volume / brain_volume
VBR = 31.2 / (150.5+120.3+110.2+90.1+140.8)
VBR = 31.2 / 611.9 = 0.051
```

---

## 兩種指標的比較

### 資料需求對比

| 需求項目 | Evans Index | VBR |
|---------|------------|-----|
| **側腦室分割** | ✅ 必需 | ✅ 必需 |
| **腦部結構分割** | ✅ 必需 | ✅ 必需 |
| **影像對齊** | ✅ 必需 | ❌ 不需要 |
| **手動測量** | ❌ 自動 | ❌ 自動 |
| **體積統計** | ❌ 不需要 | ✅ 必需 |
| **2D 定位** | ✅ 需要找最寬切片 | ❌ 不需要 |
| **3D 處理** | ❌ 單切片 | ✅ 全腦體積 |

### 程式碼依賴關係

#### Evans Index 依賴鏈
```
1. src/segmentator/run_brain_segmentation.py (分割)
   ↓
2. src/segmentator/utils.py (合併遮罩)
   ↓
3. src/segmentator/run_aligner_flirt.py (對齊)
   ↓
4. notebooks/demo_evans_index.ipynb
   - apply_euler_rotation() (旋轉校正)
   - find_best_ventricle_segment() (定位)
   - find_skull_segment() (測量顱內寬度)
   - check_hydrocephalus() (計算 EI)
```

#### VBR 依賴鏈
```
1. src/segmentator/run_brain_segmentation.py (分割+統計)
   ↓
2. statistics.json (自動生成)
   ↓
3. notebooks/demo_evans_index.ipynb
   - load_brain_data() (讀取資料)
   - calculate_brain_metrics() (計算 VBR)
```

---

## 實際執行範例

### 完整流程執行
```python
# 主程式入口
# src/pipeline_for_dcm_folder.py:79-88

if __name__ == "__main__":
    # 1. 載入 TotalSegmentator Token
    segmentator_token = os.getenv("TOTALSEGMENTATOR_TOKEN")
    
    # 2. 執行完整管線
    paths = run_pipeline_dcm_to_data_folder(
        dcm_folder_path="/path/to/dicom/",
        totalseg_token=segmentator_token,
        project_root="/path/to/MindScope"
    )
    
    # 3. 輸出檔案路徑
    # paths['brain_mask_path'] → Evans Index 分母
    # paths['ventricle_mask_path'] → Evans Index 分子
    # statistics.json → VBR 計算資料
```

### Notebook 分析
```python
# notebooks/demo_evans_index.ipynb

# Evans Index 計算
results = run_hydrocephalus_analysis(
    ventricle_nii_path=paths['aligned_ventricles'],
    skull_nii_path=paths['aligned_brain']
)
print(f"Evans Index: {results['hydrocephalus_result']}")
# 輸出: Evans Index: 0.243, 正常範圍

# VBR 計算
metrics = calculate_brain_metrics('brain_structures/statistics.json')
print(f"VBR: {metrics['VBR']}")
# 輸出: VBR: 0.0261, 正常範圍
```

---

## 關鍵檔案清單

| 檔案路徑 | 功能 | Evans Index | VBR |
|---------|------|------------|-----|
| `src/pipeline_for_dcm_folder.py` | 主流程控制 | ✅ | ✅ |
| `src/segmentator/run_brain_segmentation.py` | TotalSegmentator 呼叫 | ✅ | ✅ |
| `src/segmentator/utils.py` | 遮罩合併 | ✅ | ❌ |
| `src/segmentator/run_aligner_flirt.py` | 影像對齊 | ✅ | ❌ |
| `notebooks/demo_evans_index.ipynb` | 指標計算與視覺化 | ✅ | ✅ |
| `statistics.json` | 體積資料 | ❌ | ✅ |
| `aligning.mat` | 對齊矩陣 | ✅ | ❌ |

---

*文件版本：1.0*  
*更新日期：2025-08-12*