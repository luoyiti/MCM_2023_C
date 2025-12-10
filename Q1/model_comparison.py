"""
统一模型对比：Ensemble vs Prophet vs Chronos

在统一口径下对比三个模型：
1. SARIMA集成模型（Ensemble）
2. Prophet
3. Chronos-2 Transformer

所有模型使用相同的：
- CV设置（min_train=180, n_splits=3, horizon=30）
- 历史窗口（可配置）
- 置信区间（95% CI）
- 评估指标（RMSE/MASE/Coverage/Winkler/Pinball）

使用方法：
    cd Q1
    python model_comparison.py --input ../data.xlsx
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import warnings

# 尝试导入Prophet
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# 尝试导入Chronos
try:
    from chronos import BaseChronosPipeline
    ChronosPipeline = BaseChronosPipeline
except ImportError:
    ChronosPipeline = None

from viz_report import (
    plot_three_way_comparison,
    generate_unified_comparison_report,
)

warnings.filterwarnings('ignore')


# ======================
# 评估指标计算（统一）
# ======================

def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """计算MASE (Mean Absolute Scaled Error)"""
    errors = np.abs(y_true - y_pred)
    naive_errors = np.abs(np.diff(y_train))
    if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
        return np.nan
    return np.mean(errors) / np.mean(naive_errors)


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_lower: np.ndarray, y_upper: np.ndarray, 
                         alpha: float = 0.05) -> float:
    """计算Pinball Loss（分位数损失）"""
    quantile_lower = alpha / 2
    quantile_upper = 1 - alpha / 2
    
    errors_lower = y_true - y_lower
    pinball_lower = np.mean(np.maximum(quantile_lower * errors_lower, 
                                       (quantile_lower - 1) * errors_lower))
    
    errors_upper = y_true - y_upper
    pinball_upper = np.mean(np.maximum(quantile_upper * errors_upper, 
                                       (quantile_upper - 1) * errors_upper))
    
    return (pinball_lower + pinball_upper) / 2


def compute_winkler_score(y_true: np.ndarray, y_lower: np.ndarray, 
                         y_upper: np.ndarray, alpha: float = 0.05) -> float:
    """计算Winkler Score（区间评分）"""
    interval_width = y_upper - y_lower
    coverage_penalty = np.where(
        (y_true >= y_lower) & (y_true <= y_upper),
        0,
        (y_lower - y_true) * (y_true < y_lower) + (y_true - y_upper) * (y_true > y_upper)
    )
    return np.mean(interval_width + (2 / alpha) * coverage_penalty)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_lower: np.ndarray, y_upper: np.ndarray,
                   y_train: np.ndarray, alpha: float = 0.05) -> Dict:
    """计算完整的评估指标四件套"""
    coverage = ((y_true >= y_lower) & (y_true <= y_upper)).mean()
    mase = compute_mase(y_true, y_pred, y_train)
    pinball = compute_pinball_loss(y_true, y_pred, y_lower, y_upper, alpha)
    winkler = compute_winkler_score(y_true, y_lower, y_upper, alpha)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {
        'coverage': coverage,
        'mase': mase,
        'pinball': pinball,
        'winkler': winkler,
        'rmse': rmse,
    }


# ======================
# Prophet相关函数
# ======================

def build_prophet_frame(ts_series: pd.Series, dates: pd.Series, cap_factor: float = 1.3) -> pd.DataFrame:
    """构造Prophet输入"""
    import holidays
    cap = ts_series.max() * cap_factor
    df_p = pd.DataFrame({
        'ds': dates,
        'y': np.log1p(ts_series.values),
        'floor': 0,
        'cap': cap,
    })
    weekday = pd.DatetimeIndex(dates).dayofweek
    df_p['is_weekend'] = weekday.isin([5, 6]).astype(int)
    us_holidays = holidays.US(years=sorted({d.year for d in pd.DatetimeIndex(dates)}))
    df_p['is_holiday'] = pd.DatetimeIndex(dates).map(lambda x: int(x in us_holidays))
    return df_p


def prophet_cv(ts_raw: pd.Series, df_dates: pd.Series, n_splits: int = 3, horizon: int = 30,
               history_window: int | None = None) -> List[Dict]:
    """Prophet滚动CV"""
    if Prophet is None:
        return []
    
    results = []
    ts_use = ts_raw if history_window is None else ts_raw.iloc[-history_window:]
    dates_use = df_dates if history_window is None else df_dates.iloc[-history_window:]
    min_train = 180
    step = max(20, (len(ts_use) - min_train - horizon) // n_splits)
    
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = train_end + horizon
        if test_end > len(ts_use):
            break
        
        train = ts_use.iloc[:train_end]
        test = ts_use.iloc[train_end:test_end]
        train_df = build_prophet_frame(train, dates_use.iloc[:train_end])
        model = Prophet(
            interval_width=0.95,
            weekly_seasonality=True,
            yearly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.02,
            changepoint_range=0.2,
            growth="linear",
        )
        model.add_regressor('is_weekend')
        model.add_regressor('is_holiday')
        try:
            model.fit(train_df)
            future_dates = dates_use.iloc[train_end:test_end]
            future = build_prophet_frame(pd.Series(index=future_dates, dtype=float), future_dates)
            forecast = model.predict(future)
            pred = np.expm1(forecast["yhat"].values)
            lower = np.expm1(forecast["yhat_lower"].values)
            upper = np.expm1(forecast["yhat_upper"].values)
            
            metrics = compute_metrics(test.values, pred, lower, upper, train.values, alpha=0.05)
            results.append(metrics)
        except Exception as e:
            print(f"  Prophet fold {i} failed: {e}")
            continue
    return results


def prophet_forecast(ts_raw: pd.Series, df_dates: pd.Series, horizon: int = 60,
                     history_window: int | None = None) -> Dict:
    """Prophet预测"""
    if Prophet is None:
        return None
    
    import holidays
    ts_use = ts_raw if history_window is None else ts_raw.iloc[-history_window:]
    dates_use = df_dates if history_window is None else df_dates.iloc[-history_window:]
    train_df = build_prophet_frame(ts_use, dates_use)
    model = Prophet(
        interval_width=0.95,
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.02,
        changepoint_range=0.2,
        growth="linear",
    )
    model.add_regressor('is_weekend')
    model.add_regressor('is_holiday')
    model.fit(train_df)
    future_dates = pd.date_range(start=ts_raw.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_df = build_prophet_frame(pd.Series(index=future_dates, dtype=float), future_dates)
    forecast = model.predict(future_df)
    return {
        "ds": forecast["ds"],
        "yhat": np.expm1(forecast["yhat"]),
        "yhat_lower": np.expm1(forecast["yhat_lower"]),
        "yhat_upper": np.expm1(forecast["yhat_upper"]),
    }


# ======================
# Chronos相关函数
# ======================

def build_chronos_frame(ts_series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """构造Chronos输入"""
    return pd.DataFrame({
        'timestamp': dates,
        'item_id': 1,
        'target': ts_series.values,
    })


def chronos_cv(ts_raw: pd.Series, df_dates: pd.Series, pipeline,
               n_splits: int = 3, horizon: int = 30,
               history_window: int | None = None) -> List[Dict]:
    """Chronos滚动CV"""
    if ChronosPipeline is None or pipeline is None:
        return []
    
    results = []
    ts_use = ts_raw if history_window is None else ts_raw.iloc[-history_window:]
    dates_use = df_dates if history_window is None else df_dates.iloc[-history_window:]
    min_train = 180
    step = max(20, (len(ts_use) - min_train - horizon) // n_splits)
    
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = train_end + horizon
        if test_end > len(ts_use):
            break
        
        train = ts_use.iloc[:train_end]
        test = ts_use.iloc[train_end:test_end]
        train_dates = dates_use.iloc[:train_end]
        
        try:
            train_df = build_chronos_frame(train, train_dates)
            pred_df = pipeline.predict_df(
                train_df,
                prediction_length=len(test),
                quantile_levels=[0.025, 0.5, 0.975],
                id_column='item_id',
                timestamp_column='timestamp',
                target='target'
            )
            
            pred = pred_df['0.5'].values
            lower = pred_df['0.025'].values
            upper = pred_df['0.975'].values
            
            metrics = compute_metrics(test.values, pred, lower, upper, train.values, alpha=0.05)
            results.append(metrics)
        except Exception as e:
            print(f"  Chronos fold {i} failed: {e}")
            continue
    return results


def chronos_forecast(ts_raw: pd.Series, df_dates: pd.Series, pipeline,
                     horizon: int = 60, history_window: int | None = None) -> Dict:
    """Chronos预测"""
    if ChronosPipeline is None or pipeline is None:
        return None
    
    ts_use = ts_raw if history_window is None else ts_raw.iloc[-history_window:]
    dates_use = df_dates if history_window is None else df_dates.iloc[-history_window:]
    train_df = build_chronos_frame(ts_use, dates_use)
    
    pred_df = pipeline.predict_df(
        train_df,
        prediction_length=horizon,
        quantile_levels=[0.025, 0.5, 0.975],
        id_column='item_id',
        timestamp_column='timestamp',
        target='target'
    )
    
    if 'timestamp' in pred_df.columns:
        future_dates = pd.to_datetime(pred_df['timestamp'].values)
    else:
        future_dates = pd.date_range(
            start=ts_raw.index[-1] + pd.Timedelta(days=1), 
            periods=horizon, 
            freq="D"
        )
    
    return {
        "ds": future_dates,
        "yhat": pred_df['0.5'].values,
        "yhat_lower": pred_df['0.025'].values,
        "yhat_upper": pred_df['0.975'].values,
    }


# ======================
# 统一对比函数
# ======================

def run_unified_comparison(ts_raw: pd.Series, df_dates: pd.Series, 
                          ensemble_result: Dict, output_dir: Path,
                          history_window: int | None = None) -> Dict:
    """
    运行统一的三模型对比
    
    Returns:
        包含所有模型指标和预测结果的字典
    """
    print("\n" + "="*60)
    print("Unified Model Comparison: Ensemble vs Prophet vs Chronos")
    print("="*60)
    print("\nAll models use the same CV settings and evaluation criteria\n")
    
    results = {
        'ensemble': ensemble_result,
        'prophet': {'metrics': None, 'forecast': None},
        'chronos': {'metrics': None, 'forecast': None},
    }
    
    # Prophet comparison
    if Prophet is not None:
        print("【Prophet Model】")
        try:
            prophet_cv_results = prophet_cv(ts_raw, df_dates, n_splits=3, horizon=30,
                                           history_window=history_window)
            if prophet_cv_results:
                prophet_metrics = {
                    'rmse': np.mean([r['rmse'] for r in prophet_cv_results]),
                    'coverage': np.mean([r['coverage'] for r in prophet_cv_results]),
                    'mase': np.mean([r['mase'] for r in prophet_cv_results]),
                    'pinball': np.mean([r['pinball'] for r in prophet_cv_results]),
                    'winkler': np.mean([r['winkler'] for r in prophet_cv_results]),
                }
                results['prophet']['metrics'] = prophet_metrics
                print(f"  CV Metrics: RMSE={prophet_metrics['rmse']:.4f}, Coverage={prophet_metrics['coverage']:.3f}, "
                      f"MASE={prophet_metrics['mase']:.4f}, Winkler={prophet_metrics['winkler']:.2f}")
            
            prophet_forecast_result = prophet_forecast(ts_raw, df_dates, horizon=60,
                                                      history_window=history_window)
            results['prophet']['forecast'] = prophet_forecast_result
            print("  ✓ Forecast completed")
        except Exception as e:
            print(f"  ✗ Prophet comparison failed: {e}")
    else:
        print("【Prophet Model】Not installed, skipping")
    
    # Chronos comparison
    if ChronosPipeline is not None:
        print("\n【Chronos Model】")
        try:
            print("  Loading Chronos-2 model...")
            pipeline = ChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
            print("  ✓ Model loaded successfully")
            
            chronos_cv_results = chronos_cv(ts_raw, df_dates, pipeline, n_splits=3, horizon=30,
                                            history_window=history_window)
            if chronos_cv_results:
                chronos_metrics = {
                    'rmse': np.mean([r['rmse'] for r in chronos_cv_results]),
                    'coverage': np.mean([r['coverage'] for r in chronos_cv_results]),
                    'mase': np.mean([r['mase'] for r in chronos_cv_results]),
                    'pinball': np.mean([r['pinball'] for r in chronos_cv_results]),
                    'winkler': np.mean([r['winkler'] for r in chronos_cv_results]),
                }
                results['chronos']['metrics'] = chronos_metrics
                print(f"  CV Metrics: RMSE={chronos_metrics['rmse']:.4f}, Coverage={chronos_metrics['coverage']:.3f}, "
                      f"MASE={chronos_metrics['mase']:.4f}, Winkler={chronos_metrics['winkler']:.2f}")
            
            chronos_forecast_result = chronos_forecast(ts_raw, df_dates, pipeline, horizon=60,
                                                      history_window=history_window)
            results['chronos']['forecast'] = chronos_forecast_result
            print("  ✓ Forecast completed")
        except Exception as e:
            print(f"  ✗ Chronos comparison failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n【Chronos Model】Not installed, skipping")
    
    return results


def load_data(path: Path) -> pd.DataFrame:
    """加载数据"""
    df = pd.read_excel(path, header=1)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
    return df.sort_values('Date').reset_index(drop=True)


def load_ensemble_result(result_path: Path) -> Dict:
    """加载保存的集成预测结果"""
    with open(result_path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="统一模型对比：Ensemble vs Prophet vs Chronos")
    parser.add_argument("--input", type=Path,
                       default=Path(__file__).parent.parent.parent / "data.xlsx",
                       help="输入数据文件")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent / "results",
                       help="输出目录")
    parser.add_argument("--ensemble-result", type=Path, default=None,
                       help="保存的集成预测结果文件（.pkl）")
    parser.add_argument("--ensemble-result-dir", type=Path, default=None,
                       help="集成预测结果目录（默认: Q1/results）")
    parser.add_argument("--history-window", type=int, default=None,
                       help="(Deprecated) Use both full and 240-day windows for comparison")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("="*60)
    print("Unified Model Comparison: Ensemble vs Prophet vs Chronos")
    print("="*60)
    print("\nLoading data...")
    df_full = load_data(args.input)
    ts_raw_full = df_full.set_index('Date')['Number of  reported results'].sort_index()
    
    # 加载集成预测结果
    if args.ensemble_result:
        ensemble_result_path = args.ensemble_result
    elif args.ensemble_result_dir:
        ensemble_result_path = args.ensemble_result_dir / "ensemble_result.pkl"
    else:
        default_dir = Path(__file__).parent / "results"
        ensemble_result_path = default_dir / "ensemble_result.pkl"
    
    if not ensemble_result_path.exists():
        print(f"\nError: Ensemble result file not found: {ensemble_result_path}")
        print("Please run q1_final_clean.py first to generate ensemble results")
        return
    
    print(f"Loading ensemble result: {ensemble_result_path}")
    ensemble_result = load_ensemble_result(ensemble_result_path)
    
    # 运行两种窗口的对比
    all_results = {}
    
    # 1. Full history comparison
    print("\n" + "="*60)
    print("【Comparison 1: Full History】")
    print("="*60)
    comparison_results_full = run_unified_comparison(
        ts_raw=ts_raw_full,
        df_dates=df_full['Date'],
        ensemble_result=ensemble_result,
        output_dir=args.output_dir,
        history_window=None
    )
    all_results['full'] = comparison_results_full
    
    # Generate full history visualization
    print("\nGenerating full history comparison plot...")
    plot_three_way_comparison(
        ts_raw=ts_raw_full,
        ensemble_result=ensemble_result,
        prophet_result=comparison_results_full['prophet']['forecast'],
        chronos_result=comparison_results_full['chronos']['forecast'],
        output_dir=args.output_dir,
        history_window=None,
        suffix="_full"
    )
    
    # 2. Recent 240 days comparison
    print("\n" + "="*60)
    print("【Comparison 2: Recent 240 Days】")
    print("="*60)
    ts_raw_240 = ts_raw_full.iloc[-240:] if len(ts_raw_full) >= 240 else ts_raw_full
    df_240 = df_full[df_full['Date'].isin(ts_raw_240.index)].copy()
    
    comparison_results_240 = run_unified_comparison(
        ts_raw=ts_raw_240,
        df_dates=df_240['Date'],
        ensemble_result=ensemble_result,
        output_dir=args.output_dir,
        history_window=240
    )
    all_results['240'] = comparison_results_240
    
    # Generate 240-day visualization
    print("\nGenerating 240-day comparison plot...")
    plot_three_way_comparison(
        ts_raw=ts_raw_240,
        ensemble_result=ensemble_result,
        prophet_result=comparison_results_240['prophet']['forecast'],
        chronos_result=comparison_results_240['chronos']['forecast'],
        output_dir=args.output_dir,
        history_window=240,
        suffix="_240"
    )
    
    # Generate unified comparison report (both windows)
    print("\nGenerating unified comparison report...")
    generate_unified_comparison_report(
        all_results=all_results,
        output_dir=args.output_dir
    )
    
    # Output summary
    print("\n" + "="*60)
    print("【Unified Comparison Summary】")
    print("="*60)
    
    print("\n【Full History Comparison】")
    if comparison_results_full.get('prophet', {}).get('metrics'):
        p = comparison_results_full['prophet']['metrics']
        print(f"  Prophet: RMSE={p['rmse']:.4f}, Coverage={p['coverage']:.3f}, "
              f"MASE={p['mase']:.4f}, Winkler={p['winkler']:.2f}")
    if comparison_results_full.get('chronos', {}).get('metrics'):
        c = comparison_results_full['chronos']['metrics']
        print(f"  Chronos: RMSE={c['rmse']:.4f}, Coverage={c['coverage']:.3f}, "
              f"MASE={c['mase']:.4f}, Winkler={c['winkler']:.2f}")
    
    print("\n【Recent 240 Days Comparison】")
    if comparison_results_240.get('prophet', {}).get('metrics'):
        p = comparison_results_240['prophet']['metrics']
        print(f"  Prophet: RMSE={p['rmse']:.4f}, Coverage={p['coverage']:.3f}, "
              f"MASE={p['mase']:.4f}, Winkler={p['winkler']:.2f}")
    if comparison_results_240.get('chronos', {}).get('metrics'):
        c = comparison_results_240['chronos']['metrics']
        print(f"  Chronos: RMSE={c['rmse']:.4f}, Coverage={c['coverage']:.3f}, "
              f"MASE={c['mase']:.4f}, Winkler={c['winkler']:.2f}")
    
    print("\nNote: Ensemble metrics are available in q1_final_clean.py output")
    
    print(f"\n✓ Comparison results saved to: {args.output_dir}")
    print("  - 6_three_way_comparison_full.png (Full history three-model comparison)")
    print("  - 6_three_way_comparison_240.png (Recent 240 days three-model comparison)")
    print("  - unified_comparison_report.txt (Unified comparison report)")


if __name__ == "__main__":
    main()
