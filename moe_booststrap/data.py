"""
数据加载和预处理模块

包含数据加载、特征工程和数据集划分功能。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_PATH,
    FEATURE_COLS,
    DIST_COLS,
    N_COL,
    RANDOM_SEED,
    HOLDOUT_WORD,
    WORD_COL_CANDIDATES,
)


def load_and_split_data(
    data_path: str = None,
    holdout_word: str = None,
    test_size: float = 0.3,
    val_ratio: float = 0.5,
) -> tuple:
    """
    加载数据、预处理并划分 train/val/test（默认 70/15/15），并对特征做标准化。

    新增：将 df 中单词列等于 holdout_word（默认 eerie）的样本抽出来，
    不参与训练/验证/测试；最后用于"已训练模型"的预测与不确定性估计。
    
    参数:
        data_path: 数据文件路径，默认使用 config 中的 DATA_PATH
        holdout_word: 要保留的单词，默认使用 config 中的 HOLDOUT_WORD
        test_size: 测试集+验证集的比例
        val_ratio: 验证集在 test_size 中的比例
    
    返回:
        (X_train, X_val, X_test, P_train, P_val, P_test, 
         N_train, N_val, N_test, holdout_pack, scaler)
    """
    if data_path is None:
        data_path = DATA_PATH
    if holdout_word is None:
        holdout_word = HOLDOUT_WORD
    
    df_raw = pd.read_csv(data_path)

    # --- 1) 找到 word 列并切分 holdout ---
    word_col = None
    for c in WORD_COL_CANDIDATES:
        if c in df_raw.columns:
            word_col = c
            break

    holdout_pack = None
    if word_col is not None:
        word_series = df_raw[word_col].astype(str)
        mask_holdout = word_series.str.lower().eq(holdout_word.lower())
        if mask_holdout.any():
            df_holdout = df_raw.loc[mask_holdout].copy()
            df = df_raw.loc[~mask_holdout].copy()
            holdout_pack = {
                "word_col": word_col,
                "word": df_holdout[word_col].astype(str).tolist(),
                "df": df_holdout,
            }
            print(f"[Holdout] 已抽取 {mask_holdout.sum()} 条 '{holdout_word}' 样本，不参与 train/val/test")
        else:
            df = df_raw
    else:
        df = df_raw
        print("[Holdout] 未找到 word 列，跳过 holdout")

    # --- 2) 特征与标签预处理（用 df 的统计量，避免泄露 holdout） ---
    X = df[FEATURE_COLS].copy()
    feat_median = X.median(numeric_only=True)
    X = X.fillna(feat_median)

    P = df[DIST_COLS].copy().fillna(0.0)
    if P.to_numpy().max() > 1.5:
        P = P / 100.0
    P = P.clip(lower=0.0)
    row_sum = P.sum(axis=1).replace(0, np.nan)
    P = P.div(row_sum, axis=0).fillna(1.0 / len(DIST_COLS))

    if N_COL is not None and N_COL in df.columns:
        N = df[N_COL].fillna(df[N_COL].median()).clip(lower=1)
        N_np = N.to_numpy().astype(np.float32)
    else:
        N_np = None

    X_np = X.to_numpy().astype(np.float32)
    P_np = P.to_numpy().astype(np.float32)

    if N_np is None:
        X_train, X_tmp, P_train, P_tmp = train_test_split(
            X_np, P_np, test_size=test_size, random_state=RANDOM_SEED
        )
        X_val, X_test, P_val, P_test = train_test_split(
            X_tmp, P_tmp, test_size=val_ratio, random_state=RANDOM_SEED
        )
        N_train = N_val = N_test = None
    else:
        X_train, X_tmp, P_train, P_tmp, N_train, N_tmp = train_test_split(
            X_np, P_np, N_np, test_size=test_size, random_state=RANDOM_SEED
        )
        X_val, X_test, P_val, P_test, N_val, N_test = train_test_split(
            X_tmp, P_tmp, N_tmp, test_size=val_ratio, random_state=RANDOM_SEED
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # --- 3) Holdout 数据也按同样的预处理与 scaler 变换 ---
    if holdout_pack is not None:
        df_holdout = holdout_pack["df"]
        Xh = df_holdout[FEATURE_COLS].copy().fillna(feat_median)
        Xh_np = scaler.transform(Xh.to_numpy().astype(np.float32)).astype(np.float32)

        Ph = df_holdout[DIST_COLS].copy().fillna(0.0)
        if Ph.to_numpy().max() > 1.5:
            Ph = Ph / 100.0
        Ph = Ph.clip(lower=0.0)
        row_sum_h = Ph.sum(axis=1).replace(0, np.nan)
        Ph = Ph.div(row_sum_h, axis=0).fillna(1.0 / len(DIST_COLS))
        Ph_np = Ph.to_numpy().astype(np.float32)

        holdout_pack = {
            "word_col": holdout_pack["word_col"],
            "word": holdout_pack["word"],
            "X": Xh_np,
            "P_true": Ph_np,
        }

    return (
        X_train, X_val, X_test,
        P_train, P_val, P_test,
        N_train, N_val, N_test,
        holdout_pack,
        scaler,
    )


def make_weights_from_N(N_array: np.ndarray, mode: str = "sqrt") -> np.ndarray:
    """
    根据每个样本的参与人数 N 计算样本权重，并做均值归一化（均值=1）。
    
    参数:
        N_array: 参与人数数组
        mode: 权重计算模式，'sqrt' 或 'log1p'
    
    返回:
        归一化后的权重数组
    """
    if mode == "sqrt":
        w = np.sqrt(N_array)
    elif mode == "log1p":
        w = np.log1p(N_array)
    else:
        raise ValueError("mode must be 'sqrt' or 'log1p'")
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)
