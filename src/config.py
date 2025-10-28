#!/usr/bin/env python3
"""
專案設定管理模組
支援從環境變數、設定檔或使用預設值
"""
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """分析設定"""
    base_path: str
    occupancy_threshold: float
    max_reasonable_width: int
    output_dir: str
    screenshot_output_dir: str

    # 測量參數
    frontal_horn_ratio: float = 1/3  # 前角區域在 Z 軸的起始比例
    min_ventricle_width: int = 5
    max_skull_width_threshold: int = 500

    # Evans Index 臨床閾值
    evans_normal_threshold: float = 0.25
    evans_borderline_threshold: float = 0.30

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'AnalysisConfig':
        """從字典建立設定"""
        return cls(
            base_path=config.get('base_path', ''),
            occupancy_threshold=config.get('occupancy_threshold', 0.6),
            max_reasonable_width=config.get('max_reasonable_width', 200),
            output_dir=config.get('output_dir', 'result'),
            screenshot_output_dir=config.get('screenshot_output_dir', 'evans_slices'),
            frontal_horn_ratio=config.get('frontal_horn_ratio', 1/3),
            min_ventricle_width=config.get('min_ventricle_width', 5),
            max_skull_width_threshold=config.get('max_skull_width_threshold', 500),
            evans_normal_threshold=config.get('evans_normal_threshold', 0.25),
            evans_borderline_threshold=config.get('evans_borderline_threshold', 0.30),
        )

    @classmethod
    def load_from_file(cls, config_path: str) -> Optional['AnalysisConfig']:
        """從 JSON 檔案載入設定"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return cls.from_dict(config_data)
        except FileNotFoundError:
            print(f"⚠️ 設定檔不存在: {config_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ 設定檔格式錯誤: {e}")
            return None

    @classmethod
    def get_default(cls) -> 'AnalysisConfig':
        """取得預設設定"""
        return cls(
            base_path="/Volumes/Kuro醬の1TSSD/標記好的資料",
            occupancy_threshold=0.6,
            max_reasonable_width=200,
            output_dir="result",
            screenshot_output_dir="evans_slices"
        )

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'base_path': self.base_path,
            'occupancy_threshold': self.occupancy_threshold,
            'max_reasonable_width': self.max_reasonable_width,
            'output_dir': self.output_dir,
            'screenshot_output_dir': self.screenshot_output_dir,
            'frontal_horn_ratio': self.frontal_horn_ratio,
            'min_ventricle_width': self.min_ventricle_width,
            'max_skull_width_threshold': self.max_skull_width_threshold,
            'evans_normal_threshold': self.evans_normal_threshold,
            'evans_borderline_threshold': self.evans_borderline_threshold,
        }

    def save_to_file(self, config_path: str) -> bool:
        """儲存設定到檔案"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"✅ 設定已儲存到: {config_path}")
            return True
        except Exception as e:
            print(f"❌ 儲存設定失敗: {e}")
            return False


def load_config(config_path: Optional[str] = None) -> AnalysisConfig:
    """
    載入設定檔，按照以下優先順序：
    1. 指定的設定檔路徑
    2. 專案根目錄的 config.json
    3. 環境變數
    4. 預設值
    """
    # 1. 嘗試從指定路徑載入
    if config_path and os.path.exists(config_path):
        config = AnalysisConfig.load_from_file(config_path)
        if config:
            print(f"✅ 從設定檔載入: {config_path}")
            return config

    # 2. 嘗試從專案根目錄載入
    default_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'config.json'
    )
    if os.path.exists(default_config_path):
        config = AnalysisConfig.load_from_file(default_config_path)
        if config:
            print(f"✅ 從專案設定檔載入: {default_config_path}")
            return config

    # 3. 從環境變數載入
    env_config = {
        'base_path': os.getenv('MINDSCOPE_BASE_PATH'),
        'occupancy_threshold': float(os.getenv('MINDSCOPE_OCCUPANCY_THRESHOLD', 0.6)),
        'output_dir': os.getenv('MINDSCOPE_OUTPUT_DIR', 'result'),
    }

    if env_config['base_path']:
        print("✅ 從環境變數載入設定")
        return AnalysisConfig.from_dict(env_config)

    # 4. 使用預設值
    print("⚠️ 使用預設設定")
    return AnalysisConfig.get_default()


# 快速存取常用路徑
class Paths:
    """專案路徑管理"""

    @staticmethod
    def get_project_root() -> str:
        """取得專案根目錄"""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def get_result_dir() -> str:
        """取得結果目錄"""
        return os.path.join(Paths.get_project_root(), 'result')

    @staticmethod
    def get_screenshot_dir() -> str:
        """取得截圖目錄"""
        return os.path.join(Paths.get_project_root(), 'evans_slices')

    @staticmethod
    def ensure_dir_exists(path: str) -> str:
        """確保目錄存在，不存在則建立"""
        os.makedirs(path, exist_ok=True)
        return path
