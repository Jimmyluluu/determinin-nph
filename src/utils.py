#!/usr/bin/env python3
"""
共用工具函數模組 (向後兼容層)
此模組保留以維持向後兼容性，但實際功能已遷移至：
- data_io.data_loader: 資料載入功能
- core.validation: 驗證與 Evans Index 計算功能
"""
# 從新模組匯入函數以保持向後兼容
from data_io.data_loader import find_available_datasets, check_prelabeled_data_paths
from core.validation import (
    calculate_evans_index,
    load_hydrocephalus_reference,
    validate_results_against_reference
)

# 匯出所有函數
__all__ = [
    'find_available_datasets',
    'check_prelabeled_data_paths',
    'calculate_evans_index',
    'load_hydrocephalus_reference',
    'validate_results_against_reference'
]