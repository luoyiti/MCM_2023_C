"""
shared 模块：项目公共工具

包含:
- config: 路径配置
- data_loader: 数据加载工具
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    BACKUP_DIR,
    ORIGINAL_DATA,
    PROCESSED_DATA,
    RESULTS_DIR,
    PICTURES_DIR,
    TASK1_RESULTS,
    TASK1_PICTURES,
    TASK2_RESULTS,
    TASK2_PICTURES,
    MODELS_DIR,
)

from .data_loader import load_original_data, load_processed_data

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'BACKUP_DIR',
    'ORIGINAL_DATA',
    'PROCESSED_DATA',
    'RESULTS_DIR',
    'PICTURES_DIR',
    'TASK1_RESULTS',
    'TASK1_PICTURES',
    'TASK2_RESULTS',
    'TASK2_PICTURES',
    'MODELS_DIR',
    'load_original_data',
    'load_processed_data',
]
