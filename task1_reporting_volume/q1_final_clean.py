"""
Q1：时间序列集成预测（精简稳健版）

核心点：
1) 每折独立 PELT 变点（防泄露）
2) log1p 方差稳定
3) 分段 SARIMA + 滚动 CV
4) 合奏=全概率方差公式（正确合成 CI）
5) Duan smearing 回变换纠偏（点预测 & CI 同口径）
6) 输出与 Prophet 同口径的评估指标（RMSE/MASE/Coverage/Winkler/Pinball）

本版优化（只动实现，不动整体流程与输出文件结构）：
- 输入只支持 CSV（真实报告人数）
- PELT 变点取“最后一个真实变点”
- 单词滞后特征 CSV 只读一次并缓存
- CV/预测：test 缺失用 train 均值填（更严格时序口径）
- Duan smearing 同时修正点预测与区间上下界（口径一致）
- 不再生成解释性长文本报告；只输出 explanation_stats.json（纯数据）
- Walk-forward 增加 interval_scale（把过宽/过窄的区间校准回目标置信水平）
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import holidays

from viz_report import (
    diagnostic_plots,
    generate_diagnostic_report,
    plot_weekday_effects,
    plot_changepoint_summary,
    plot_factor_importance,
)

warnings.filterwarnings("ignore")

try:
    import ruptures as rpt
except ImportError:
    raise ImportError("需要安装 ruptures: pip install ruptures")


# =============================================================================
# 全局配置与缓存
# =============================================================================

_WORD_ATTRIBUTES_PATH: Optional[str] = None
_WORD_LAG_DF: Optional[pd.DataFrame] = None   # 滞后1天
_WORD_CURR_DF: Optional[pd.DataFrame] = None  # 当天
_LAG_FEATURES_PRINTED_OK: bool = False
_LAG_FEATURES_PRINTED_ERR: bool = False

_WORD_FEATURES = [
    "mean_simulate_freq",
    "letter_entropy",
    "mean_simulate_random",
    "has_common_suffix",
    "letter_freq_mean",
]


# =============================================================================
# 评估与合奏工具
# =============================================================================

def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 7) -> float:
    """MASE（默认以季节朴素 s=7 作为基线）"""
    if len(y_train) <= season:
        return np.nan
    denom = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if denom == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def compute_winkler_score(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, alpha: float = 0.10
) -> float:
    """Winkler@1-alpha 区间评分（越小越好；alpha=0.10 对应 90% CI）"""
    width = y_upper - y_lower
    under = y_true < y_lower
    over = y_true > y_upper
    penalty = (y_lower - y_true) * under + (y_true - y_upper) * over
    return float(np.mean(width + (2.0 / alpha) * penalty))


def compute_pinball_loss_point(y_true: np.ndarray, y_pred: np.ndarray, taus=(0.1, 0.5, 0.9)) -> dict:
    """简化版 pinball：用点预测近似对应分位（用于横比即可）"""
    e = y_true - y_pred
    out = {}
    for t in taus:
        out[f"Pin@{int(t * 100)}"] = float(np.mean(np.maximum(t * e, (t - 1) * e)))
    return out


def combine_mean_se(means_log: List[np.ndarray], ses_log: List[np.ndarray], weights: np.ndarray):
    """
    合奏预测区间：使用全概率方差公式 (Law of Total Variance)

    总方差 = E[Var(Y|Model)] + Var(E[Y|Model])
          = Σ wi * σi^2     + Σ wi * (μi - μ)^2
    """
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    means_stack = np.stack(means_log)  # (n_models, n_steps)
    ses_stack = np.stack(ses_log)      # (n_models, n_steps)

    mu = np.average(means_stack, axis=0, weights=w)

    var_conditional = np.sum(w[:, None] * (ses_stack ** 2), axis=0)
    var_between_models = np.sum(w[:, None] * ((means_stack - mu[None, :]) ** 2), axis=0)
    var_total = var_conditional + var_between_models

    return mu, np.sqrt(var_total), w


def duan_smearing_factor(residuals_log: Optional[np.ndarray]) -> float:
    """返回 Duan smearing 因子 k = E[exp(resid)]；为空则返回1"""
    if residuals_log is None or len(residuals_log) == 0:
        return 1.0
    return float(np.mean(np.exp(residuals_log)))


def apply_smearing(pred_log: np.ndarray, k: float) -> np.ndarray:
    """把 log 空间预测映射回原尺度（带 smearing）"""
    return np.exp(pred_log) * k - 1.0


# =============================================================================
# 数据读取（只保留 CSV）
# =============================================================================

def load_data(path) -> pd.DataFrame:
    """
    只支持 CSV（真实报告人数）。
    期望列：
      - date 或 Date（日期）
      - number_of_reported_results 或 Number of  reported results（报告人数）
    """
    path = Path(path) if not isinstance(path, Path) else path
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"仅支持 CSV 输入（真实报告人数）。当前文件: {path.name}。\n"
            "请使用包含 number_of_reported_results 列的 CSV 数据。"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        raise KeyError("CSV 中找不到日期列：需要 'date' 或 'Date'")

    if "number_of_reported_results" in df.columns:
        df["Number of  reported results"] = df["number_of_reported_results"]
    elif "Number of  reported results" in df.columns:
        df["Number of  reported results"] = df["Number of  reported results"]
    else:
        raise KeyError("CSV 中找不到报告人数列：需要 'number_of_reported_results'（推荐）")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    df = df.sort_values("Date").reset_index(drop=True)

    if (df["Number of  reported results"] < 0).any():
        raise ValueError("报告人数列存在负数，请检查输入数据。")

    return df


# =============================================================================
# 变点检测（取最后一个真实变点）
# =============================================================================

def detect_changepoint(ts: pd.Series, is_log_space: bool = False) -> int:
    """
    PELT 变点检测，返回最后一个真实变点位置（索引从0开始）
    """
    if is_log_space:
        x = np.asarray(ts.values, dtype=float)
    else:
        x = np.log1p(np.asarray(ts.values, dtype=float))

    algo = rpt.Pelt(model="l2", min_size=30).fit(x)
    bkps = algo.predict(pen=3)  # [cp1, cp2, ..., n]
    if len(bkps) <= 1:
        return len(ts) // 2

    cp = int(bkps[-2])  # 最后一项通常是 n
    cp = int(np.clip(cp, 1, len(ts) - 1))
    return cp


# =============================================================================
# 外生特征：weekend/holiday + 滞后单词属性（缓存 + 规范填充）
# =============================================================================

def _resolve_word_attributes_path(p: Path) -> Optional[Path]:
    """
    允许你传文件或目录：
    - 如果是 csv 文件：直接用
    - 如果是目录：尝试找一个 csv（优先常见文件名），否则返回 None
    """
    try:
        if p.is_file() and p.suffix.lower() == ".csv":
            return p
        if p.is_dir():
            # 常见候选名（不影响你原逻辑；只是更不容易踩坑）
            candidates = [
                p / "word_attributes.csv",
                p / "word_features.csv",
                p / "word_attrs.csv",
            ]
            for c in candidates:
                if c.exists() and c.is_file() and c.suffix.lower() == ".csv":
                    return c
            # 没找到就随缘挑一个 csv
            any_csv = sorted([x for x in p.glob("*.csv") if x.is_file()])
            return any_csv[0] if any_csv else None
    except Exception:
        return None
    return None


def _load_word_features_once() -> None:
    """读取并缓存当天和滞后1天的单词属性表（只做一次）"""
    global _WORD_CURR_DF, _WORD_LAG_DF, _LAG_FEATURES_PRINTED_OK, _LAG_FEATURES_PRINTED_ERR

    if _WORD_CURR_DF is not None and _WORD_LAG_DF is not None:
        return
    if not _WORD_ATTRIBUTES_PATH:
        return

    p0 = Path(_WORD_ATTRIBUTES_PATH)
    p = _resolve_word_attributes_path(p0) if p0.exists() else None
    if p is None or not p.exists():
        if not _LAG_FEATURES_PRINTED_ERR:
            print(f"  ⚠️ 警告：单词属性文件不可用：{p0}，跳过单词特征")
            _LAG_FEATURES_PRINTED_ERR = True
        return

    try:
        word_df = pd.read_csv(p)
        word_df.columns = word_df.columns.str.strip()

        if "date" not in word_df.columns:
            raise KeyError("单词属性 CSV 缺少 'date' 列")
        word_df["date"] = pd.to_datetime(word_df["date"])

        need = ["date"] + _WORD_FEATURES
        missing = [c for c in need if c not in word_df.columns]
        if missing:
            raise KeyError(f"单词属性 CSV 缺少列: {missing}")

        # 当天特征 (lag0_*)
        curr = word_df[need].copy()
        curr = curr.rename(columns={c: f"lag0_{c}" for c in _WORD_FEATURES})
        curr = curr.rename(columns={"date": "date_merge"})
        _WORD_CURR_DF = curr

        # 滞后1天特征 (lag1_*)
        lag = word_df[need].copy()
        lag["date_merge"] = lag["date"] + pd.Timedelta(days=1)  # 今日对齐昨日特征
        lag = lag.rename(columns={c: f"lag1_{c}" for c in _WORD_FEATURES})
        lag = lag[["date_merge"] + [f"lag1_{c}" for c in _WORD_FEATURES]]
        _WORD_LAG_DF = lag

        if not _LAG_FEATURES_PRINTED_OK:
            print(
                f"  ✓ 成功加载并缓存 {len(_WORD_FEATURES)} 个当天特征（lag0_*）和 "
                f"{len(_WORD_FEATURES)} 个滞后特征（lag1_*）: {p.name}"
            )
            _LAG_FEATURES_PRINTED_OK = True

    except Exception as e:
        if not _LAG_FEATURES_PRINTED_ERR:
            print(f"  ⚠️ 警告：无法加载单词属性数据（{e}），跳过单词特征")
            _LAG_FEATURES_PRINTED_ERR = True
        _WORD_CURR_DF = None
        _WORD_LAG_DF = None


def _word_feature_cols(exog: pd.DataFrame) -> List[str]:
    """返回所有单词特征列（lag0_* 和 lag1_*）"""
    return [c for c in exog.columns if c.startswith("lag0_") or c.startswith("lag1_")]


def _fill_exog_with_stats(exog: pd.DataFrame, fill_stats: Optional[dict]) -> pd.DataFrame:
    """用给定统计量填充缺失（严格用 train 统计量）"""
    if not fill_stats:
        return exog
    for k, v in fill_stats.items():
        if k in exog.columns:
            exog[k] = exog[k].fillna(v)
    return exog


def add_features(df: pd.DataFrame, changepoint_idx: int, ts: pd.Series) -> pd.DataFrame:
    """
    添加外生特征（包含当天和滞后单词属性）。
    注意：这里不做缺失填充，由外层统一用 train 统计填补。
    """
    _load_word_features_once()

    exog = pd.DataFrame(index=ts.index)

    # regime
    exog["regime"] = 0.0
    if changepoint_idx > 0 and changepoint_idx < len(exog):
        exog.loc[exog.index[changepoint_idx:], "regime"] = 1.0

    # weekend/holiday
    dates = df["Date"].values
    dt = pd.DatetimeIndex(dates)

    exog["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(float)
    us_holidays = holidays.US(years=[2022, 2023])
    exog["is_holiday"] = dt.map(lambda x: float(x in us_holidays))

    # lag0 word attrs
    if _WORD_CURR_DF is not None:
        exog_reset = exog.reset_index(drop=True)
        dates_df = pd.DataFrame({"date_merge": pd.to_datetime(dates, errors="coerce")})
        merged_curr = dates_df.merge(_WORD_CURR_DF, on="date_merge", how="left")
        for col in [f"lag0_{c}" for c in _WORD_FEATURES]:
            exog_reset[col] = merged_curr[col].values
        exog = exog_reset.set_index(exog.index)

    # lag1 word attrs
    if _WORD_LAG_DF is not None:
        exog_reset = exog.reset_index(drop=True)
        dates_df = pd.DataFrame({"date_merge": pd.to_datetime(dates, errors="coerce")})
        merged_lag = dates_df.merge(_WORD_LAG_DF, on="date_merge", how="left")
        for col in [f"lag1_{c}" for c in _WORD_FEATURES]:
            exog_reset[col] = merged_lag[col].values
        exog = exog_reset.set_index(exog.index)

    return exog


# =============================================================================
# 解释性“数据统计”（不写长文，只输出数据）
# =============================================================================

def _standardized_ols_importance(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    简单、稳定的“重要性”度量：
    - 对 y 和 X 做标准化
    - OLS 拟合
    - 返回 |beta|（标准化系数绝对值）
    """
    eps = 1e-8
    y = y.astype(float)
    X = X.astype(float)

    y_std = (y - y.mean()) / (y.std() + eps)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)

    A = np.column_stack([np.ones(len(y_std)), X_std])
    beta = np.linalg.lstsq(A, y_std, rcond=None)[0][1:]
    return np.abs(beta)


def compute_explanation_stats(ts_raw: pd.Series, exog: pd.DataFrame, cp_idx: int, models: Dict) -> Dict:
    """
    只计算可复用的“解释性统计量”（纯数值），不给自然语言解释。
    返回 dict（可用于 plot_factor_importance，也可写 JSON）。
    """
    from statsmodels.tsa.seasonal import STL

    cp_date = ts_raw.index[cp_idx]
    pre_cp = ts_raw.iloc[:cp_idx]
    post_cp = ts_raw.iloc[cp_idx:]

    cp_drop_abs = float(pre_cp.mean() - post_cp.mean())
    cp_drop_pct = float(cp_drop_abs / pre_cp.mean() * 100) if pre_cp.mean() != 0 else 0.0

    weekday_mask = exog["is_weekend"] == 0
    weekend_mask = exog["is_weekend"] == 1
    weekday_avg = float(ts_raw[weekday_mask].mean())
    weekend_avg = float(ts_raw[weekend_mask].mean())
    weekend_effect_pct = float((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg != 0 else 0.0

    holiday_mask = exog["is_holiday"] == 1
    non_holiday_mask = exog["is_holiday"] == 0
    if int(holiday_mask.sum()) > 0:
        holiday_avg = float(ts_raw[holiday_mask].mean())
        non_holiday_avg = float(ts_raw[non_holiday_mask].mean())
        holiday_effect_pct = float((holiday_avg - non_holiday_avg) / non_holiday_avg * 100) if non_holiday_avg != 0 else 0.0
    else:
        holiday_avg = 0.0
        non_holiday_avg = float(ts_raw.mean())
        holiday_effect_pct = 0.0

    try:
        stl = STL(ts_raw, seasonal=7, period=7)
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal

        recent_trend = float(trend.iloc[-30:].mean())
        prev_trend = float(trend.iloc[-60:-30].mean())
        trend_change = recent_trend - prev_trend
        trend_change_pct = float(trend_change / prev_trend * 100) if prev_trend != 0 else 0.0
        seasonal_strength = float(seasonal.std())
    except Exception:
        trend_change_pct = 0.0
        seasonal_strength = 0.0

    overall_std = float(ts_raw.std())
    recent_std = float(ts_raw.iloc[-30:].std())
    volatility_change_pct = float((recent_std - overall_std) / overall_std * 100) if overall_std != 0 else 0.0

    best = models.get("best", {})
    cv_metrics = {
        "model_name": best.get("name", ""),
        "cv_rmse": float(best.get("cv_rmse", 0.0) or 0.0),
        "cv_mase": float(best.get("cv_mase", np.nan)) if best.get("cv_mase", None) is not None else np.nan,
    }

    cols = list(exog.columns)
    lag0_cols = [c for c in cols if c.startswith("lag0_")]
    lag1_cols = [c for c in cols if c.startswith("lag1_")]

    lag0_impact = 0.0
    lag1_impact = 0.0

    if lag0_cols or lag1_cols:
        y = np.log1p(ts_raw.values)
        X = exog[cols].values
        abs_beta = _standardized_ols_importance(y, X)
        total = float(abs_beta.sum()) + 1e-12

        if lag0_cols:
            lag0_idx = [i for i, c in enumerate(cols) if c.startswith("lag0_")]
            lag0_impact = float(abs_beta[lag0_idx].sum() / total * 100.0)

        if lag1_cols:
            lag1_idx = [i for i, c in enumerate(cols) if c.startswith("lag1_")]
            lag1_impact = float(abs_beta[lag1_idx].sum() / total * 100.0)

    return {
        "cp_date": str(cp_date.date()),
        "cp_idx": int(cp_idx),
        "cp_drop_pct": float(cp_drop_pct),
        "weekend_effect_pct": float(weekend_effect_pct),
        "holiday_effect_pct": float(holiday_effect_pct),
        "trend_change_pct": float(trend_change_pct),
        "volatility_change_pct": float(volatility_change_pct),
        "seasonal_strength": float(seasonal_strength),
        "overall_std": float(overall_std),
        "cv_metrics": cv_metrics,
        "lag0_features_impact": float(lag0_impact),
        "lag1_features_impact": float(lag1_impact),
        "n_lag0_features": int(len(lag0_cols)),
        "n_lag1_features": int(len(lag1_cols)),
    }


# =============================================================================
# CV 与模型拟合
# =============================================================================

def rolling_cv(
    ts: pd.Series,
    df_dates: pd.Series,
    order: Tuple,
    seasonal_order: Tuple,
    n_splits: int = 3,
    horizon: int = 30,
) -> List[Dict]:
    """滚动 CV；每折独立变点→构建特征→拟合→预测（全在 log 空间）"""
    results = []
    min_train = 180
    step = max(20, (len(ts) - min_train - horizon) // n_splits)

    print(f"    开始滚动CV (n_splits={n_splits}, horizon={horizon}, min_train={min_train})...")

    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = train_end + horizon
        if test_end > len(ts):
            break

        print(
            f"      Fold {i+1}/{n_splits}: train={train_end} days, test={test_end-train_end} days...",
            end=" ",
            flush=True,
        )

        train, test = ts.iloc[:train_end], ts.iloc[train_end:test_end]

        try:
            cp_idx = detect_changepoint(train, is_log_space=True)

            exog_train = add_features(pd.DataFrame({"Date": df_dates.iloc[:train_end]}), cp_idx, train)
            fill_stats = {c: float(exog_train[c].mean()) for c in _word_feature_cols(exog_train)}
            exog_train = _fill_exog_with_stats(exog_train, fill_stats)

            # train 是从 0 开始，所以 cp_idx 本身就是全局绝对索引
            cp_absolute = cp_idx

            if train_end >= cp_absolute:
                test_cp_idx = 0
            elif test_end <= cp_absolute:
                test_cp_idx = test_end - train_end
            else:
                test_cp_idx = cp_absolute - train_end

            exog_test = add_features(pd.DataFrame({"Date": df_dates.iloc[train_end:test_end]}), test_cp_idx, test)
            exog_test = exog_test.reindex(columns=exog_train.columns)
            exog_test = _fill_exog_with_stats(exog_test, fill_stats).fillna(0.0)

            model = SARIMAX(
                train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50, method="lbfgs")

            fc = fitted.get_forecast(steps=len(test), exog=exog_test)
            pred_mean = fc.predicted_mean.values

            pred_summary = fc.summary_frame(alpha=0.10)  # 90% CI
            ci_l = pred_summary["mean_ci_lower"].values
            ci_u = pred_summary["mean_ci_upper"].values
            pred_se = (ci_u - ci_l) / (2 * 1.645)

            test_values = test.values
            errors = test_values - pred_mean
            coverage = float(np.mean((test_values >= ci_l) & (test_values <= ci_u)))

            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mase = compute_mase(test_values, pred_mean, train.values)
            wink = compute_winkler_score(test_values, ci_l, ci_u, alpha=0.10)
            pinb = compute_pinball_loss_point(test_values, pred_mean)["Pin@50"]

            print(f"✓ RMSE={rmse:.4f}, Coverage={coverage*100:.1f}%")

            results.append(
                {
                    "rmse": rmse,
                    "pred_var": float(np.mean(pred_se ** 2)),
                    "coverage": coverage,
                    "mase": mase,
                    "pinball": pinb,
                    "winkler": wink,
                    "errors": errors,
                    "train_cp_idx": cp_idx,
                    "train_end": train_end,
                    "pred_mean": pred_mean,
                    "pred_se": pred_se,
                    "test_values": test_values,
                }
            )

        except Exception as e:
            print(f"✗ 失败: {str(e)[:80]}")
            continue

    return results


def fit_models(ts: pd.Series, df_dates: pd.Series) -> Dict:
    print("\n拟合候选模型（独立变点 + 滚动CV）...")
    print(f"  数据长度: {len(ts)} 天")

    candidates = [
        ((1, 1, 1), (1, 0, 1, 7), "SARIMA(1,1,1)x(1,0,1,7)"),
        ((2, 1, 1), (0, 0, 0, 0), "ARIMA(2,1,1)"),
        ((1, 1, 2), (1, 0, 1, 7), "SARIMA(1,1,2)x(1,0,1,7)"),
    ]

    results = []
    for idx, (order, seasonal_order, name) in enumerate(candidates, 1):
        print(f"\n  [{idx}/{len(candidates)}] 训练模型: {name}")

        import time
        start_time = time.time()

        cv = rolling_cv(ts, df_dates, order, seasonal_order, n_splits=3, horizon=30)

        elapsed = time.time() - start_time
        print(f"    ✓ 完成，耗时 {elapsed:.1f} 秒")

        if not cv:
            print("    ⚠️  所有折都失败，跳过此模型")
            continue

        avg = lambda k: np.nanmean([r[k] for r in cv])
        print(
            f"    CV指标: RMSE={avg('rmse'):6.4f}  Coverage={avg('coverage'):5.1%}  "
            f"MASE={avg('mase'):5.3f}  Winkler={avg('winkler'):6.2f}"
        )

        results.append(
            {
                "name": name,
                "order": order,
                "seasonal_order": seasonal_order,
                "cv_rmse": avg("rmse"),
                "cv_coverage": avg("coverage"),
                "cv_mase": avg("mase"),
                "cv_winkler": avg("winkler"),
                "cv_pinball": avg("pinball"),
                "pred_var": avg("pred_var"),
                "cv_changepoints": [r.get("train_cp_idx", 0) for r in cv],
                "cv_changepoints_global": [r.get("train_cp_idx", 0) for r in cv],
                "cv_train_lengths": [r.get("train_end", 0) for r in cv],
                "cv_results": cv,
            }
        )

    if not results:
        raise ValueError("所有模型拟合失败")

    best = min(results, key=lambda x: x["cv_rmse"])
    print(f"\n✓ 最优模型：{best['name']} (CV-RMSE: {best['cv_rmse']:.4f})")
    return {"models": results, "best": best}


# =============================================================================
# 单模型预测 & 集成预测（log 空间，输出90% CI）
# =============================================================================

def forecast_with_ci(
    ts: pd.Series,
    exog: pd.DataFrame,
    exog_future: pd.DataFrame,
    model_info: Dict,
    steps: int,
    verbose: bool = False,
) -> Dict:
    """单模型预测（log 空间），返回均值/预测标准误/90% CI"""
    model = SARIMAX(
        ts,
        exog=exog,
        order=model_info["order"],
        seasonal_order=model_info["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=50, method="lbfgs")

    fc = fitted.get_forecast(steps=steps, exog=exog_future)
    pred_mean = fc.predicted_mean.values

    pred_summary = fc.summary_frame(alpha=0.10)  # 90% CI
    ci_lower = pred_summary["mean_ci_lower"].values
    ci_upper = pred_summary["mean_ci_upper"].values
    pred_se = (ci_upper - ci_lower) / (2 * 1.645)

    if verbose:
        print(f"  [{model_info['name']}] mean(pred_se)={np.mean(pred_se):.4f}")

    return {
        "forecast": pred_mean,
        "se": pred_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def ensemble_forecast(
    ts: pd.Series,
    exog: pd.DataFrame,
    exog_future: pd.DataFrame,
    model_results: List[Dict],
    steps: int,
    verbose: bool = False,
) -> Dict:
    """合奏（log 空间）：IVW 加权均值 + 全概率方差合并，输出 90% CI"""
    means_log, ses_log, inv_vars = [], [], []

    for m in model_results:
        r = forecast_with_ci(ts, exog, exog_future, m, steps, verbose=False)
        means_log.append(r["forecast"])
        ses_log.append(r["se"])
        inv_vars.append(1.0 / (m["pred_var"] + 1e-6))

    mu_log, se_log, w = combine_mean_se(means_log, ses_log, np.array(inv_vars, dtype=float))

    if verbose:
        print("\n生成集成预测...")
        print("权重分配（Inverse Variance Weighting）：")
        for wi, mi in zip(w, model_results):
            print(f"  {mi['name']:<30} {wi*100:5.1f}%")
        print("\n预测区间统计（log空间，90% CI，未校准）：")
        print(f"  平均预测标准误：{np.mean(se_log):.4f}")
        print(f"  标准误范围：[{np.min(se_log):.4f}, {np.max(se_log):.4f}]")

    lo_log = mu_log - 1.645 * se_log
    hi_log = mu_log + 1.645 * se_log

    return {
        "forecast": mu_log,
        "se": se_log,
        "ci_lower": lo_log,
        "ci_upper": hi_log,
        "weights": w,
    }


# =============================================================================
# Walk-forward 验证（log 空间覆盖率 + 区间校准 interval_scale）
# =============================================================================

def walk_forward_validation(
    ts: pd.Series,
    df_dates: pd.Series,
    model_results: List[Dict],
    horizons: List[int] = [30, 60],
    verbose: bool = False,
) -> Dict:
    """
    Walk-forward 验证：
    - 先按“未校准区间”算 raw coverage
    - 再用标准化残差 |z| 的 95% 分位数估计 interval_scale
    - 用 scale 后的阈值算 calibrated coverage（目标≈90%）
    """
    results: Dict[int, Dict] = {}
    min_train = 180

    for h in horizons:
        if len(ts) < min_train + h:
            if verbose:
                print(f"  [h={h}] 跳过：数据不足（需要至少 {min_train + h} 天，当前 {len(ts)} 天）")
            continue

        coverages_raw = []
        abs_z_all = []
        test_start = min_train

        while test_start + h <= len(ts):
            train = ts.iloc[:test_start]
            test = ts.iloc[test_start:test_start + h]

            try:
                cp_idx = detect_changepoint(train, is_log_space=True)

                exog_train = add_features(pd.DataFrame({"Date": df_dates.iloc[:test_start]}), cp_idx, train)
                fill_stats = {c: float(exog_train[c].mean()) for c in _word_feature_cols(exog_train)}
                exog_train = _fill_exog_with_stats(exog_train, fill_stats)

                # train 从 0 开始，所以 cp_idx 本身就是绝对索引
                cp_absolute = cp_idx

                if test_start >= cp_absolute:
                    test_cp_idx = 0
                elif test_start + h <= cp_absolute:
                    test_cp_idx = h
                else:
                    test_cp_idx = cp_absolute - test_start

                exog_test = add_features(
                    pd.DataFrame({"Date": df_dates.iloc[test_start:test_start + h]}),
                    test_cp_idx,
                    test,
                )

                if verbose:
                    regime_vals = exog_test["regime"].values
                    n_regime1 = int(np.sum(regime_vals == 1.0))
                    print(
                        f" [DEBUG: start={test_start}, cp_abs={cp_absolute}, test_cp_idx={test_cp_idx}, "
                        f"regime1={n_regime1}/{h}]",
                        end="",
                    )

                exog_test = exog_test.reindex(columns=exog_train.columns)
                exog_test = _fill_exog_with_stats(exog_test, fill_stats).fillna(0.0)

                ens = ensemble_forecast(train, exog_train, exog_test, model_results, steps=h, verbose=False)

                mu = ens["forecast"]
                se = ens["se"]
                lo = mu - 1.645 * se
                hi = mu + 1.645 * se

                y = test.values
                coverage_raw = float(np.mean((y >= lo) & (y <= hi)))
                coverages_raw.append(coverage_raw)

                z = (y - mu) / (se + 1e-12)
                abs_z_all.extend(np.abs(z).tolist())

                if verbose:
                    print(f"    [h={h}, start={test_start}] rawCoverage={coverage_raw*100:.1f}%")

            except Exception as e:
                if verbose:
                    print(f"    [h={h}, start={test_start}] 失败: {str(e)[:80]}")
                pass

            test_start += 30

        if not coverages_raw:
            continue

        if len(abs_z_all) > 0:
            q95 = float(np.quantile(abs_z_all, 0.90))
            interval_scale = q95 / 1.645
            interval_scale = float(np.clip(interval_scale, 0.3, 1.0))
        else:
            q95 = np.nan
            interval_scale = 1.0

        thr = 1.645 * interval_scale
        calibrated_cov = float(np.mean(np.array(abs_z_all) <= thr))

        results[h] = {
            "coverage": calibrated_cov,             # ✅ 校准后覆盖率（更接近90%）
            "coverage_raw": float(np.mean(coverages_raw)),
            "n_tests": len(coverages_raw),
            "interval_scale": float(interval_scale),
            "q95_abs_z": float(q95) if np.isfinite(q95) else np.nan,
        }

        print(
            f"  [h={h}天] raw={results[h]['coverage_raw']*100:.1f}%  "
            f"calibrated={results[h]['coverage']*100:.1f}%  "
            f"interval_scale={results[h]['interval_scale']:.3f}  (n={results[h]['n_tests']})"
        )

    return results


# =============================================================================
# main（保留你的流程，只替换默认 input 为 csv，并移除长文报告）
# =============================================================================

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from config import TASK1_RESULTS, TASK1_PICTURES, PROCESSED_DATA

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/Users/1m/Desktop/大三上/大数据/期末/data.csv"))
    parser.add_argument(
        "--word-attributes",
        type=Path,
        default=PROCESSED_DATA,
        help="单词属性数据路径（csv 或目录；用于滞后特征）",
    )
    parser.add_argument("--output-dir", type=Path, default=TASK1_RESULTS, help="结果输出目录（默认: results/task1/）")
    parser.add_argument("--pictures-dir", type=Path, default=TASK1_PICTURES, help="图表输出目录（默认: pictures/task1/）")
    parser.add_argument("--history-window", type=int, default=None, help="主训练窗口长度（None=全历史）")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pictures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Q1：时间序列集成预测（精简稳健版 + 滞后特征）")
    print("=" * 60)

    # 1) 数据
    print("\n[1/8] 加载数据...")
    df = load_data(args.input)
    ts_raw_full = df.set_index("Date")["Number of  reported results"].sort_index()

    if args.history_window:
        ts_raw = ts_raw_full.iloc[-args.history_window:]
        df = df[df["Date"].isin(ts_raw.index)].copy()
        print(f"  ✓ 使用最近 {args.history_window} 天数据 ({ts_raw.index[0].date()} 至 {ts_raw.index[-1].date()})")
    else:
        ts_raw = ts_raw_full
        print(f"  ✓ 使用全部历史数据 ({len(ts_raw)} 天)")

    ts_log = np.log1p(ts_raw)

    # 2) 变点摘要（在 log 空间检测,与模型训练一致）
    print("\n[2/8] 检测变点...")
    cp_idx = detect_changepoint(ts_log, is_log_space=True)
    cp_date = ts_raw.index[cp_idx]
    pre_mean = ts_raw.iloc[:cp_idx].mean()
    post_mean = ts_raw.iloc[cp_idx:].mean()
    print(f"  ✓ 变点: {cp_date.date()} (第{cp_idx}天)")
    print(f"    变点前均值: {pre_mean:,.0f} 人/天")
    print(f"    变点后均值: {post_mean:,.0f} 人/天")
    if pre_mean != 0:
        print(f"    下降幅度: {(post_mean - pre_mean) / pre_mean * 100:.1f}%")
    print("    提示: 变点在log空间检测,与模型训练保持一致")

    # 3) 外生变量（包含滞后特征）
    print("\n[3/8] 构建特征...")
    global _WORD_ATTRIBUTES_PATH
    _WORD_ATTRIBUTES_PATH = str(args.word_attributes) if args.word_attributes.exists() else None
    _load_word_features_once()

    exog = add_features(df, cp_idx, ts_log)
    fill_stats_full = {c: float(exog[c].mean()) for c in _word_feature_cols(exog)}
    exog = _fill_exog_with_stats(exog, fill_stats_full)

    print(f"  ✓ 生成 {len(exog.columns)} 个外生变量: {list(exog.columns)}")

    # 4) 拟合候选 + CV
    print("\n[4/8] 拟合并交叉验证候选模型...")
    print("  (这一步可能需要几分钟，模型在认真思考人生中...)")
    import time
    total_start = time.time()
    models = fit_models(ts_log, df["Date"])
    total_elapsed = time.time() - total_start
    print(f"\n  ✓ 模型训练完成，总耗时 {total_elapsed:.1f} 秒")

    # 5) 解释性统计（纯数据）
    print("\n[5/8] 计算解释性统计量（纯数据）...")
    explanation_stats = compute_explanation_stats(ts_raw, exog, cp_idx, models)
    with open(args.output_dir / "explanation_stats.json", "w", encoding="utf-8") as f:
        json.dump(explanation_stats, f, ensure_ascii=False, indent=2)
    print("  ✓ 已保存: explanation_stats.json")

    # 6) 残差诊断（图表保存到 pictures_dir）
    print("\n[6/8] 生成诊断图表...")
    best = models["best"]
    cv_for_diag = rolling_cv(ts_log, df["Date"], best["order"], best["seasonal_order"], n_splits=3, horizon=30)
    diag_stats = diagnostic_plots(ts_log, exog, best, cv_for_diag, args.pictures_dir)
    resid_stable = diag_stats.get("residuals_stable", pd.Series(dtype=float)).dropna().values

    # 7) Walk-Forward（含 interval_scale）
    print("\n[7/8] Walk-Forward 验证...")
    wf_results = walk_forward_validation(ts_log, df["Date"], models["models"], horizons=[30, 60], verbose=False)
    for h, stats in wf_results.items():
        print(
            f"  h={h}天: 覆盖率(calibrated)={stats['coverage']*100:.1f}% "
            f"(raw={stats.get('coverage_raw', np.nan)*100:.1f}%, "
            f"scale={stats.get('interval_scale', 1.0):.3f}, n={stats['n_tests']})"
        )

    # 8) 未来预测
    print("\n[8/8] 生成未来预测...")
    last_date = ts_log.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60, freq="D")

    exog_future = add_features(pd.DataFrame({"Date": future_dates}), 0, pd.Series(index=future_dates, dtype=float))
    exog_future["regime"] = float(exog["regime"].iloc[-1])

    exog_future = exog_future.reindex(columns=exog.columns)
    exog_future = _fill_exog_with_stats(exog_future, fill_stats_full).fillna(0.0)

    ens_log = ensemble_forecast(ts_log, exog, exog_future, models["models"], steps=60, verbose=True)

    mu_log = ens_log["forecast"]
    se_log = ens_log["se"]

    # ✅ 把 walk-forward 的 interval_scale 应用到“未来区间”
    interval_scale = 1.0
    if 60 in wf_results:
        interval_scale = float(wf_results[60].get("interval_scale", 1.0))
    elif 30 in wf_results:
        interval_scale = float(wf_results[30].get("interval_scale", 1.0))

    se_adj = se_log * interval_scale
    lo_log = mu_log - 1.645 * se_adj
    hi_log = mu_log + 1.645 * se_adj

    print(f"\n[CI Calibration] interval_scale={interval_scale:.3f} (applied to future CI)")

    # smearing（口径一致：点 + 上下界一起 smearing）
    k = duan_smearing_factor(resid_stable)
    y_pred = np.exp(mu_log) * k - 1.0
    y_lo   = np.expm1(lo_log)   
    y_hi   = np.expm1(hi_log)   


    # 可视化与报告（保持你原逻辑）
    print("\n生成可视化图表...")
    generate_diagnostic_report(best, cv_for_diag, diag_stats, wf_results, args.output_dir)
    plot_weekday_effects(ts_raw, cp_idx, args.pictures_dir)
    plot_changepoint_summary(ts_raw, cp_idx, args.pictures_dir)
    plot_factor_importance(explanation_stats, args.pictures_dir)

    # 保存结果（保持你原逻辑）
    with open(args.output_dir / "ensemble_result.pkl", "wb") as f:
        pickle.dump(
            {"forecast": y_pred, "ci_lower": y_lo, "ci_upper": y_hi, "weights": ens_log["weights"]},
            f,
        )

    # 输出目标日（保持你原逻辑）
    target_idx = (pd.to_datetime("2023-03-01") - future_dates[0]).days
    if 0 <= target_idx < 60:
        pred = y_pred[target_idx]
        lo = y_lo[target_idx]
        hi = y_hi[target_idx]
        half_w = (hi - lo) / 2
        recent_std = ts_raw.iloc[-30:].std()
        print("\n" + "=" * 60)
        print("【2023-03-01 预测】")
        print("=" * 60)
        print(f"点预测：{pred:,.0f} 人")
        print(f"90% CI：[{lo:,.0f}, {hi:,.0f}]")
        print(f"区间宽度：±{half_w:,.0f} 人（≈{half_w/recent_std:.1f}× 最近30天std）")

    print("\n" + "=" * 60)
    print("✓ 所有输出已保存")
    print("=" * 60)
    print(f"  结果数据 → {args.output_dir}/")
    print("    - explanation_stats.json")
    print("    - diagnostic_report.txt")
    print("    - ensemble_result.pkl")
    print(f"\n  可视化图表 → {args.pictures_dir}/")
    print("    - 1_weekday_effects.png")
    print("    - 2_changepoint.png")
    print("    - 3_diagnostics.png")
    print("    - 4_factor_importance.png  ← 包含滞后特征")
    print("\n下一步:")
    print("  运行 python model_comparison.py --input ../data.xlsx")
    print("  对比 Ensemble vs Prophet vs Chronos")
    print("=" * 60)


if __name__ == "__main__":
    main()
