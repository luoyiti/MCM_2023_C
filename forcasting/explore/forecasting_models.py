"""
模块化预测模型库
包含多种回归模型的训练、评估和可视化函数
支持的模型：Lasso, Ridge, ElasticNet, MLP, RandomForest, TabNet
"""

import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

warnings.filterwarnings("ignore")

# ============================================================================
# 全局配置
# ============================================================================

# 默认特征列
DEFAULT_FEATURE_COLS = [
    "字母频率特征_weighted_reduced",
    "位置特征_PLS_reduced",
    "仿真模拟特征_weighted_reduced",
    "强化学习特征_weighted_reduced",
    "Zipf-value",
    "feedback_entropy",
    "letter_entropy",
    "max_consecutive_vowels",
    "semantic_distance",
]

DEFAULT_TARGET_COL = "autoencoder_value"
RANDOM_STATE = 42


def setup_plot_style(font_family: str = 'Heiti TC'):
    """设置绘图样式"""
    plt.rcParams['font.family'] = font_family
    sns.set_style("whitegrid")
    sns.set_palette("husl")


# ============================================================================
# 数据加载与预处理
# ============================================================================

def load_data(
    train_path: str = None,
    test_path: str = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = DEFAULT_TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    加载训练集和测试集数据
    
    Parameters:
    -----------
    train_path : str
        训练集文件路径
    test_path : str
        测试集文件路径
    feature_cols : list, optional
        特征列名列表，默认使用DEFAULT_FEATURE_COLS
    target_col : str
        目标列名
        
    Returns:
    --------
    X_train, y_train, X_test, y_test
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    if train_path is None:
        train_path = config.TRAIN_DATA
    if test_path is None:
        test_path = config.TEST_DATA
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"训练集样本数: {len(df_train)}")
    print(f"测试集样本数: {len(df_test)}")
    
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].copy()
    
    return X_train, y_train, X_test, y_test


def preprocess_data(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    数据预处理：缺失值填充
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        训练集特征
    X_test : pd.DataFrame
        测试集特征
        
    Returns:
    --------
    X_train, X_test : 处理后的数据
    """
    print(f"训练集缺失值: {X_train.isnull().sum().sum()}")
    print(f"测试集缺失值: {X_test.isnull().sum().sum()}")
    
    if X_train.isnull().sum().sum() > 0:
        X_train = X_train.fillna(X_train.median(numeric_only=True))
    if X_test.isnull().sum().sum() > 0:
        X_test = X_test.fillna(X_train.median(numeric_only=True))
    
    return X_train, X_test


# ============================================================================
# 模型评估指标计算
# ============================================================================

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    计算回归评估指标
    
    Returns:
    --------
    dict : 包含 r2, rmse, mae 的字典
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def print_metrics(train_metrics: Dict, test_metrics: Dict, model_name: str = ""):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print(f"{model_name} 评估结果")
    print("=" * 60)
    print(f"\n训练集:")
    print(f"  R²  = {train_metrics['r2']:.4f}")
    print(f"  RMSE = {train_metrics['rmse']:.4f}")
    print(f"  MAE  = {train_metrics['mae']:.4f}")
    print(f"\n测试集:")
    print(f"  R²  = {test_metrics['r2']:.4f}")
    print(f"  RMSE = {test_metrics['rmse']:.4f}")
    print(f"  MAE  = {test_metrics['mae']:.4f}")


# ============================================================================
# LASSO 回归模型
# ============================================================================

def train_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv_splits: int = 5,
    verbose: int = 1
) -> Tuple[Pipeline, Dict]:
    """
    训练LASSO回归模型
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        训练特征
    y_train : pd.Series
        训练目标
    param_grid : dict, optional
        超参数网格
    cv_splits : int
        交叉验证折数
    verbose : int
        日志详细程度
        
    Returns:
    --------
    best_model : Pipeline
        最优模型
    search_results : dict
        搜索结果信息
    """
    if param_grid is None:
        param_grid = {
            "lasso__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        }
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(random_state=RANDOM_STATE, max_iter=5000)),
    ])
    
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    
    search_results = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_
    }
    
    print(f"\n最优参数: {search.best_params_}")
    print(f"最优交叉验证R²: {search.best_score_:.4f}")
    
    return search.best_estimator_, search_results


def get_lasso_coefficients(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    """获取LASSO模型系数"""
    lasso_model = model.named_steps["lasso"]
    coef = lasso_model.coef_
    
    df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)
    
    n_zero = np.sum(coef == 0)
    print(f"\n被L1正则化压缩为零的特征数量: {n_zero}/{len(feature_cols)}")
    
    return df


# ============================================================================
# Ridge 回归模型
# ============================================================================

def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv_splits: int = 5,
    verbose: int = 1
) -> Tuple[Pipeline, Dict]:
    """
    训练Ridge回归模型
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        训练特征
    y_train : pd.Series
        训练目标
    param_grid : dict, optional
        超参数网格
    cv_splits : int
        交叉验证折数
        
    Returns:
    --------
    best_model : Pipeline
        最优模型
    search_results : dict
        搜索结果信息
    """
    if param_grid is None:
        param_grid = {
            "ridge__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            "ridge__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        }
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=RANDOM_STATE)),
    ])
    
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    
    search_results = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_
    }
    
    print(f"\n最优参数: {search.best_params_}")
    print(f"最优交叉验证R²: {search.best_score_:.4f}")
    
    return search.best_estimator_, search_results


def get_ridge_coefficients(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    """获取Ridge模型系数"""
    ridge_model = model.named_steps["ridge"]
    coef = ridge_model.coef_
    
    return pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)


# ============================================================================
# Elastic Net 回归模型
# ============================================================================

def train_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv_splits: int = 5,
    verbose: int = 1
) -> Tuple[Pipeline, Dict]:
    """
    训练Elastic Net回归模型
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        训练特征
    y_train : pd.Series
        训练目标
    param_grid : dict, optional
        超参数网格
    cv_splits : int
        交叉验证折数
        
    Returns:
    --------
    best_model : Pipeline
        最优模型
    search_results : dict
        搜索结果信息
    """
    if param_grid is None:
        param_grid = {
            "elasticnet__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("elasticnet", ElasticNet(random_state=RANDOM_STATE, max_iter=2000)),
    ])
    
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    
    search_results = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_
    }
    
    print(f"\n最优参数: {search.best_params_}")
    print(f"最优交叉验证R²: {search.best_score_:.4f}")
    
    return search.best_estimator_, search_results


def get_elasticnet_coefficients(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    """获取Elastic Net模型系数"""
    en_model = model.named_steps["elasticnet"]
    coef = en_model.coef_
    
    df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)
    
    n_zero = np.sum(coef == 0)
    print(f"\n被L1正则化压缩为零的特征数量: {n_zero}/{len(feature_cols)}")
    
    return df


# ============================================================================
# MLP 回归模型
# ============================================================================

def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv_splits: int = 5,
    verbose: int = 1
) -> Tuple[Pipeline, Dict]:
    """
    训练MLP回归模型
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        训练特征
    y_train : pd.Series
        训练目标
    param_grid : dict, optional
        超参数网格
    cv_splits : int
        交叉验证折数
        
    Returns:
    --------
    best_model : Pipeline
        最优模型
    search_results : dict
        搜索结果信息
    """
    if param_grid is None:
        param_grid = {
            "mlp__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3],
            "mlp__activation": ["relu", "tanh"],
        }
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=5000,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )),
    ])
    
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    print("（MLP训练可能需要几分钟时间，请耐心等待...）")
    
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    
    search_results = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_
    }
    
    print(f"\n最优参数: {search.best_params_}")
    print(f"最优交叉验证R²: {search.best_score_:.4f}")
    
    return search.best_estimator_, search_results


# ============================================================================
# Random Forest 回归模型
# ============================================================================

def train_randomforest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv_splits: int = 5,
    verbose: int = 1
) -> Tuple[Pipeline, Dict]:
    """
    训练Random Forest回归模型
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        训练特征
    y_train : pd.Series
        训练目标
    param_grid : dict, optional
        超参数网格
    cv_splits : int
        交叉验证折数
        
    Returns:
    --------
    best_model : Pipeline
        最优模型
    search_results : dict
        搜索结果信息
    """
    if param_grid is None:
        param_grid = {
            "rf__n_estimators": [100, 200, 300],
            "rf__max_depth": [10, 20, 30, None],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__max_features": ["sqrt", "log2"],
        }
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    print("（Random Forest训练可能需要几分钟，请耐心等待...）")
    
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    
    search_results = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_
    }
    
    print(f"\n最优参数: {search.best_params_}")
    print(f"最优交叉验证R²: {search.best_score_:.4f}")
    
    return search.best_estimator_, search_results


def get_randomforest_feature_importance(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    """获取Random Forest模型特征重要性"""
    rf_model = model.named_steps["rf"]
    importance = rf_model.feature_importances_
    
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)


# ============================================================================
# TabNet 回归模型
# ============================================================================

# TabNet特征列（更多特征）
TABNET_FEATURE_COLS = [
    'Zipf-value',
    'letter_entropy',
    'feedback_entropy',
    'max_consecutive_vowels',
    'letter_freq_mean',
    'scrabble_score',
    'has_common_suffix',
    'num_rare_letters',
    'position_rarity',
    'positional_freq_min',
    'hamming_neighbors',
    'keyboard_distance',
    'semantic_distance',
    '1_try_simulate_random', '2_try_simulate_random', '3_try_simulate_random',
    '4_try_simulate_random', '5_try_simulate_random', '6_try_simulate_random', '7_try_simulate_random',
    '1_try_simulate_freq', '2_try_simulate_freq', '3_try_simulate_freq',
    '4_try_simulate_freq', '5_try_simulate_freq', '6_try_simulate_freq', '7_try_simulate_freq',
    '1_try_simulate_entropy', '2_try_simulate_entropy', '3_try_simulate_entropy',
    '4_try_simulate_entropy', '5_try_simulate_entropy', '6_try_simulate_entropy', '7_try_simulate_entropy',
    'rl_1_try_low_training', 'rl_2_try_low_training', 'rl_3_try_low_training',
    'rl_4_try_low_training', 'rl_5_try_low_training', 'rl_6_try_low_training', 'rl_7_try_low_training',
    'rl_1_try_high_training', 'rl_2_try_high_training', 'rl_3_try_high_training',
    'rl_4_try_high_training', 'rl_5_try_high_training', 'rl_6_try_high_training', 'rl_7_try_high_training',
    'rl_1_try_little_training', 'rl_2_try_little_training', 'rl_3_try_little_training',
    'rl_4_try_little_training', 'rl_5_try_little_training', 'rl_6_try_little_training', 'rl_7_try_little_training',
]


def load_tabnet_data(
    data_path: str = "data/mcm_processed_data.csv",
    feature_cols: Optional[List[str]] = None,
    target_col: str = DEFAULT_TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    为TabNet加载数据
    
    Returns:
    --------
    X, y, available_features
    """
    if feature_cols is None:
        feature_cols = TABNET_FEATURE_COLS
    
    df = pd.read_csv(data_path)
    print(f"数据集加载: {df.shape[0]} 行, {df.shape[1]} 列")
    
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"警告: 缺少特征: {missing_features}")
    
    print(f"使用 {len(available_features)} 个特征")
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    return X, y, available_features


def preprocess_tabnet_data(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    TabNet数据预处理
    
    Returns:
    --------
    X_np, y_np : numpy数组
    """
    X = X.fillna(X.median(numeric_only=True))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_np = X_scaled.astype(np.float32)
    y_np = y.to_numpy().reshape(-1, 1).astype(np.float32)
    
    print(f"特征矩阵形状: {X_np.shape}")
    print(f"目标向量形状: {y_np.shape}")
    
    return X_np, y_np


def create_tabnet_model():
    """创建TabNet回归模型"""
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        
        model = TabNetRegressor(
            n_d=24,
            n_a=24,
            n_steps=4,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-4,
            optimizer_params=dict(lr=1e-2, weight_decay=1e-4),
            mask_type="entmax",
            seed=RANDOM_STATE,
            verbose=0
        )
        return model
    except ImportError:
        raise ImportError("请先安装 pytorch-tabnet: pip install pytorch-tabnet")


def train_tabnet_cv(
    X_np: np.ndarray,
    y_np: np.ndarray,
    n_splits: int = 5,
    max_epochs: int = 1000,
    patience: int = 100
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray, Dict]:
    """
    使用交叉验证训练TabNet
    
    Parameters:
    -----------
    X_np : np.ndarray
        特征矩阵
    y_np : np.ndarray
        目标向量
    n_splits : int
        交叉验证折数
        
    Returns:
    --------
    all_y_test, all_y_pred, fold_metrics, avg_feature_importance, cv_summary
    """
    from sklearn.model_selection import train_test_split
    
    print(f"\n使用 {n_splits} 折交叉验证训练TabNet")
    print("=" * 60)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    rmse_list, mae_list, r2_list = [], [], []
    all_y_test, all_y_pred = [], []
    fold_metrics = []
    feature_importances = []
    
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_np), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y_np[tr_idx], y_np[te_idx]
        
        X_tr2, X_va, y_tr2, y_va = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE
        )
        
        model = create_tabnet_model()
        
        model.fit(
            X_train=X_tr2, y_train=y_tr2,
            eval_set=[(X_va, y_va)],
            eval_metric=["rmse"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=min(64, X_tr2.shape[0]),
            virtual_batch_size=min(32, max(1, X_tr2.shape[0] // 4)),
            num_workers=0,
            drop_last=False
        )
        
        pred = model.predict(X_te)
        
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        r2 = r2_score(y_te, pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        
        all_y_test.extend(y_te.flatten())
        all_y_pred.extend(pred.flatten())
        
        fold_metrics.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
        feature_importances.append(model.feature_importances_)
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("交叉验证汇总")
    print("=" * 60)
    print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"MAE:  {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"R²:   {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    
    avg_feature_importance = np.mean(feature_importances, axis=0)
    
    cv_summary = {
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'r2_mean': np.mean(r2_list),
        'r2_std': np.std(r2_list),
    }
    
    return (np.array(all_y_test), np.array(all_y_pred), 
            fold_metrics, avg_feature_importance, cv_summary)


# ============================================================================
# 交叉验证评估
# ============================================================================

def cross_validate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5
) -> Tuple[float, float, List[float]]:
    """
    对已训练的模型进行交叉验证评估
    
    Returns:
    --------
    cv_mean, cv_std, cv_scores
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for train_idx, val_idx in kfold.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 克隆模型参数
        model.fit(X_cv_train, y_cv_train)
        y_cv_pred = model.predict(X_cv_val)
        cv_r2 = r2_score(y_cv_val, y_cv_pred)
        cv_scores.append(cv_r2)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"交叉验证 R²: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return cv_mean, cv_std, cv_scores


# ============================================================================
# 可视化函数
# ============================================================================

def plot_prediction_scatter(
    y_train, y_train_pred, y_test, y_test_pred,
    train_metrics: Dict, test_metrics: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制预测散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 训练集
    axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    mn, mx = min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True autoencoder_value', fontsize=12)
    axes[0].set_ylabel('Predicted autoencoder_value', fontsize=12)
    axes[0].set_title(f"Training Set\nR²={train_metrics['r2']:.4f}, RMSE={train_metrics['rmse']:.4f}", 
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 测试集
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='green')
    mn, mx = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    axes[1].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('True autoencoder_value', fontsize=12)
    axes[1].set_ylabel('Predicted autoencoder_value', fontsize=12)
    axes[1].set_title(f"Test Set\nR²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}", 
                      fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals(
    y_train, y_train_pred, y_test, y_test_pred,
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制残差图"""
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(y_train_pred, train_residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0].axhline(0, color='r', ls='--', lw=2)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Training Set Residuals', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test_pred, test_residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='green')
    axes[1].axhline(0, color='r', ls='--', lw=2)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Test Set Residuals', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_distribution_comparison(
    y_train, y_train_pred, y_test, y_test_pred,
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制分布对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(y_train, bins=30, alpha=0.6, label='True', color='blue', edgecolor='black')
    axes[0].hist(y_train_pred, bins=30, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
    axes[0].set_xlabel('autoencoder_value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Training Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].hist(y_test, bins=20, alpha=0.6, label='True', color='blue', edgecolor='black')
    axes[1].hist(y_test_pred, bins=20, alpha=0.6, label='Predicted', color='green', edgecolor='black')
    axes[1].set_xlabel('autoencoder_value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Test Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(
    train_metrics: Dict, test_metrics: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制指标对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['R²', 'RMSE', 'MAE']
    train_vals = [train_metrics['r2'], train_metrics['rmse'], train_metrics['mae']]
    test_vals = [test_metrics['r2'], test_metrics['rmse'], test_metrics['mae']]
    
    x = np.arange(len(metrics))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, train_vals, w, label='Train', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + w/2, test_vals, w, label='Test', edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Metrics Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h, f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制特征重要性图（支持coefficient和importance两种列名）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 自动检测使用哪个列名
    if 'coefficient' in feature_importance_df.columns:
        value_col = 'coefficient'
        xlabel = 'Coefficient'
    elif 'importance' in feature_importance_df.columns:
        value_col = 'importance'
        xlabel = 'Importance'
    else:
        raise ValueError("DataFrame必须包含'coefficient'或'importance'列")
    
    values = feature_importance_df[value_col].values
    features = feature_importance_df['feature'].values
    
    # 对于系数可以有负值（红色），对于重要性都是正值（蓝色）
    if value_col == 'coefficient':
        colors = ['red' if c < 0 else 'blue' for c in values]
    else:
        colors = ['steelblue'] * len(values)
    
    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7, edgecolor='black')
    
    if value_col == 'coefficient':
        ax.axvline(0, color='black', lw=0.8)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for b, v in zip(bars, values):
        if abs(v) > 0.001:
            ax.text(b.get_width(), b.get_y() + b.get_height()/2, f'{v:.3f}', 
                    ha='left' if v > 0 else 'right', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_error_distribution(
    y_train, y_train_pred, y_test, y_test_pred,
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制误差分布图"""
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(train_residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0, color='r', ls='--', lw=2)
    axes[0].set_title(f"Train Error Dist\nMean={np.mean(train_residuals):.4f}, Std={np.std(train_residuals):.4f}", 
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].hist(test_residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(0, color='r', ls='--', lw=2)
    axes[1].set_title(f"Test Error Dist\nMean={np.mean(test_residuals):.4f}, Std={np.std(test_residuals):.4f}", 
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_cv_fold_metrics(
    fold_metrics: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制交叉验证各折指标"""
    folds = [m['fold'] for m in fold_metrics]
    rmse_vals = [m['rmse'] for m in fold_metrics]
    mae_vals = [m['mae'] for m in fold_metrics]
    r2_vals = [m['r2'] for m in fold_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].bar(folds, rmse_vals, color='coral', edgecolor='black')
    axes[0].axhline(y=np.mean(rmse_vals), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(rmse_vals):.4f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Fold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(folds, mae_vals, color='skyblue', edgecolor='black')
    axes[1].axhline(y=np.mean(mae_vals), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(mae_vals):.4f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE by Fold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].bar(folds, r2_vals, color='lightgreen', edgecolor='black')
    axes[2].axhline(y=np.mean(r2_vals), color='green', linestyle='--',
                    label=f'Mean: {np.mean(r2_vals):.4f}')
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² by Fold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_tabnet_feature_importance(
    feature_names: List[str],
    importance: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None,
    show: bool = True
):
    """绘制TabNet特征重要性"""
    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]
    
    n_features = min(top_n, len(feature_names))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(n_features)
    ax.barh(y_pos, sorted_importance[:n_features], align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names[:n_features])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'TabNet Feature Importance (Top {n_features})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return sorted_names, sorted_importance


# ============================================================================
# 完整训练流程封装
# ============================================================================

def train_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    output_dir: Optional[str] = None,
    save_plots: bool = True,
    show_plots: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    统一的模型训练与评估流程
    
    Parameters:
    -----------
    model_name : str
        模型名称: 'lasso', 'ridge', 'elasticnet', 'mlp'
    X_train, y_train : 训练数据
    X_test, y_test : 测试数据
    feature_cols : list
        特征列名
    output_dir : str, optional
        输出目录
    save_plots : bool
        是否保存图表
    show_plots : bool
        是否显示图表
        
    Returns:
    --------
    dict : 包含模型、预测结果、指标等
    """
    model_name_lower = model_name.lower()
    
    # 创建输出目录
    if output_dir is None:
        output_dir = f"{model_name_lower}_results"
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} 模型训练与评估")
    print("=" * 60)
    
    # 训练模型
    if model_name_lower == 'lasso':
        model, search_results = train_lasso(X_train, y_train, **kwargs)
        coef_df = get_lasso_coefficients(model, feature_cols)
    elif model_name_lower == 'ridge':
        model, search_results = train_ridge(X_train, y_train, **kwargs)
        coef_df = get_ridge_coefficients(model, feature_cols)
    elif model_name_lower == 'elasticnet':
        model, search_results = train_elasticnet(X_train, y_train, **kwargs)
        coef_df = get_elasticnet_coefficients(model, feature_cols)
    elif model_name_lower == 'mlp':
        model, search_results = train_mlp(X_train, y_train, **kwargs)
        coef_df = None
    elif model_name_lower == 'randomforest':
        model, search_results = train_randomforest(X_train, y_train, **kwargs)
        coef_df = get_randomforest_feature_importance(model, feature_cols)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算指标
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # 打印指标
    print_metrics(train_metrics, test_metrics, model_name.upper())
    
    # 绘图
    save_prefix = f"{output_dir}/" if save_plots else None
    
    plot_prediction_scatter(
        y_train, y_train_pred, y_test, y_test_pred,
        train_metrics, test_metrics,
        save_path=f"{save_prefix}1_prediction_scatter.png" if save_plots else None,
        show=show_plots
    )
    
    plot_residuals(
        y_train, y_train_pred, y_test, y_test_pred,
        save_path=f"{save_prefix}2_residuals.png" if save_plots else None,
        show=show_plots
    )
    
    plot_distribution_comparison(
        y_train, y_train_pred, y_test, y_test_pred,
        save_path=f"{save_prefix}3_distribution_comparison.png" if save_plots else None,
        show=show_plots
    )
    
    plot_metrics_comparison(
        train_metrics, test_metrics,
        save_path=f"{save_prefix}4_metrics_comparison.png" if save_plots else None,
        show=show_plots
    )
    
    if coef_df is not None:
        plot_feature_importance(
            coef_df,
            title=f"Feature Importance ({model_name.upper()})",
            save_path=f"{save_prefix}5_feature_importance.png" if save_plots else None,
            show=show_plots
        )
    
    plot_error_distribution(
        y_train, y_train_pred, y_test, y_test_pred,
        save_path=f"{save_prefix}6_error_distribution.png" if save_plots else None,
        show=show_plots
    )
    
    return {
        'model': model,
        'model_name': model_name,
        'search_results': search_results,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'coefficients': coef_df,
    }


def compare_models(results_list: List[Dict], save_path: Optional[str] = None, show: bool = True):
    """
    比较多个模型的性能
    
    Parameters:
    -----------
    results_list : list of dict
        各模型的训练结果
    """
    model_names = [r['model_name'] for r in results_list]
    train_r2 = [r['train_metrics']['r2'] for r in results_list]
    test_r2 = [r['test_metrics']['r2'] for r in results_list]
    train_rmse = [r['train_metrics']['rmse'] for r in results_list]
    test_rmse = [r['test_metrics']['rmse'] for r in results_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(model_names))
    w = 0.35
    
    # R² 对比
    axes[0].bar(x - w/2, train_r2, w, label='Train', alpha=0.8, edgecolor='black')
    axes[0].bar(x + w/2, test_r2, w, label='Test', alpha=0.8, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² Comparison', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMSE 对比
    axes[1].bar(x - w/2, train_rmse, w, label='Train', alpha=0.8, edgecolor='black')
    axes[1].bar(x + w/2, test_rmse, w, label='Test', alpha=0.8, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # 打印汇总表
    print("\n" + "=" * 70)
    print("模型性能汇总")
    print("=" * 70)
    print(f"{'Model':<15} {'Train R²':<12} {'Test R²':<12} {'Train RMSE':<12} {'Test RMSE':<12}")
    print("-" * 70)
    for r in results_list:
        print(f"{r['model_name']:<15} "
              f"{r['train_metrics']['r2']:<12.4f} "
              f"{r['test_metrics']['r2']:<12.4f} "
              f"{r['train_metrics']['rmse']:<12.4f} "
              f"{r['test_metrics']['rmse']:<12.4f}")


def save_predictions(
    results: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str
):
    """保存预测结果到CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = X_train.copy()
    train_df['y_true'] = y_train.values
    train_df['y_pred'] = results['train_predictions']
    train_df.to_csv(f"{output_dir}/train_predictions.csv", index=False)
    
    test_df = X_test.copy()
    test_df['y_true'] = y_test.values
    test_df['y_pred'] = results['test_predictions']
    test_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)
    
    print(f"预测结果已保存到 {output_dir}/")


def generate_report(
    results: Dict,
    output_path: str
):
    """生成文本报告"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{results['model_name'].upper()} 回归模型 - 预测报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    lines.append("\n## 最优超参数")
    lines.append("-" * 40)
    for k, v in results['search_results']['best_params'].items():
        lines.append(f"{k}: {v}")
    
    lines.append(f"\n交叉验证 R²: {results['search_results']['best_cv_score']:.4f}")
    
    lines.append("\n## 训练集指标")
    lines.append("-" * 40)
    for k, v in results['train_metrics'].items():
        lines.append(f"{k.upper()}: {v:.4f}")
    
    lines.append("\n## 测试集指标")
    lines.append("-" * 40)
    for k, v in results['test_metrics'].items():
        lines.append(f"{k.upper()}: {v:.4f}")
    
    if results['coefficients'] is not None:
        coef_df = results['coefficients']
        # 自动检测使用coefficient还是importance列
        if 'coefficient' in coef_df.columns:
            value_col = 'coefficient'
            lines.append("\n## 特征重要性（按系数绝对值排序）")
        elif 'importance' in coef_df.columns:
            value_col = 'importance'
            lines.append("\n## 特征重要性（按重要性排序）")
        else:
            value_col = None
        
        if value_col:
            lines.append("-" * 40)
            for _, row in coef_df.iterrows():
                lines.append(f"{row['feature']:<40} {row[value_col]:>10.4f}")
    
    lines.append("\n" + "=" * 70)
    lines.append("报告结束")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到 {output_path}")
    
    return report
