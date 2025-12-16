"""
项目路径配置模块

统一管理所有路径，避免硬编码
"""

from pathlib import Path

# ============ 项目根目录 ============
PROJECT_ROOT = Path(__file__).parent.parent

# ============ 数据路径 ============
DATA_DIR = PROJECT_ROOT / "data"
BACKUP_DIR = PROJECT_ROOT / "backups"

# 具体数据文件
ORIGINAL_DATA = BACKUP_DIR / "2023_MCM_Problem_C_Data.xlsx"
PROCESSED_DATA = DATA_DIR / "mcm_processed_data.csv"

# ============ 输出路径 ============
RESULTS_DIR = PROJECT_ROOT / "results"
PICTURES_DIR = PROJECT_ROOT / "pictures"

# 任务1路径
TASK1_RESULTS = RESULTS_DIR / "task1"
TASK1_PICTURES = PICTURES_DIR / "task1"

# 任务2路径
TASK2_RESULTS = RESULTS_DIR / "task2"
TASK2_PICTURES = PICTURES_DIR / "task2"

# ============ 模型路径 ============
MODELS_DIR = PROJECT_ROOT / "models"

# ============ 确保目录存在 ============
def ensure_dirs():
    """创建所有必要的输出目录"""
    for dir_path in [
        TASK1_RESULTS, TASK1_PICTURES,
        TASK2_RESULTS, TASK2_PICTURES,
        MODELS_DIR
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


# 自动创建目录
ensure_dirs()


# ============ 打印路径信息（调试用）============
if __name__ == "__main__":
    print("=" * 60)
    print("项目路径配置")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"\n数据:")
    print(f"  原始数据: {ORIGINAL_DATA}")
    print(f"  处理数据: {PROCESSED_DATA}")
    print(f"\n输出:")
    print(f"  结果: {RESULTS_DIR}")
    print(f"  图片: {PICTURES_DIR}")
    print(f"\n任务1:")
    print(f"  结果: {TASK1_RESULTS}")
    print(f"  图片: {TASK1_PICTURES}")
    print(f"\n任务2:")
    print(f"  结果: {TASK2_RESULTS}")
    print(f"  图片: {TASK2_PICTURES}")
    print(f"\n模型: {MODELS_DIR}")
    print("=" * 60)
