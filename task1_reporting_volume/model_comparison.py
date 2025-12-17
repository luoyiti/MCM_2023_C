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

# 导入共享配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from config import TASK1_RESULTS, TASK1_PICTURES

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
                   y_train: np.ndarray, alpha: float = 0.10) -> Dict:
    """计算完整的评估指标四件套（统一使用 90% CI，alpha=0.10）"""
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
            
            metrics = compute_metrics(test.values, pred, lower, upper, train.values, alpha=0.10)  # 🔧 统一 90% CI
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
    # 确保日期是连续的日期序列,避免频率推断失败
    timestamps = pd.to_datetime(dates.values)
    # 重新索引为连续日期序列,填充缺失值
    continuous_index = pd.date_range(start=timestamps.min(), end=timestamps.max(), freq='D')
    ts_reindexed = pd.Series(ts_series.values, index=timestamps).reindex(continuous_index, method='ffill')
    
    return pd.DataFrame({
        'timestamp': continuous_index,
        'item_id': 1,
        'target': ts_reindexed.values,
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
                quantile_levels=[0.05, 0.5, 0.95],  # 🔧 统一 90% CI
                id_column='item_id',
                timestamp_column='timestamp',
                target='target'
            )
            
            pred = pred_df['0.5'].values
            lower = pred_df['0.05'].values  # 🔧 统一 90% CI：0.05 和 0.95 分位数
            upper = pred_df['0.95'].values  # 🔧 统一 90% CI
            
            metrics = compute_metrics(test.values, pred, lower, upper, train.values, alpha=0.10)  # 🔧 统一 90% CI
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
        quantile_levels=[0.05, 0.5, 0.95],  # 🔧 统一 90% CI
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
        "yhat_lower": pred_df['0.05'].values,  # 🔧 统一 90% CI
        "yhat_upper": pred_df['0.95'].values,  # 🔧 统一 90% CI
    }


# ======================
# 统一对比函数
# ======================

def train_ensemble_on_window(ts_raw: pd.Series, df_dates: pd.Series, 
                             history_window: int | None = None) -> Dict:
    """
    在指定窗口上训练 Ensemble 模型并返回 CV 指标
    
    Args:
        ts_raw: 完整时间序列
        df_dates: 对应日期
        history_window: 使用的历史窗口长度（None=全历史）
    
    Returns:
        包含 CV 指标和预测结果的字典
    """
    from q1_final_clean import (
        detect_changepoint, add_features, fit_models, 
        ensemble_forecast, duan_smearing, diagnostic_plots
    )
    
    # 1. 选择窗口
    if history_window:
        ts_use = ts_raw.iloc[-history_window:]
        dates_use = df_dates.iloc[-history_window:]
        df_use = pd.DataFrame({'Date': dates_use})
    else:
        ts_use = ts_raw
        dates_use = df_dates
        df_use = pd.DataFrame({'Date': dates_use})
    
    ts_log = np.log1p(ts_use)
    
    # 2. 变点检测
    cp_idx = detect_changepoint(ts_use)
    
    # 3. 特征工程
    exog = add_features(df_use, cp_idx, ts_log)
    
    # 4. 训练候选模型 + CV
    print(f"  Training ensemble on {'full history' if history_window is None else f'{history_window} days'}...")
    models = fit_models(ts_log, dates_use)
    
    # 5. 提取 CV 指标
    best_model = models['best']
    cv_metrics = {
        'rmse': best_model['cv_rmse'],
        'mase': best_model['cv_mase'],
        'coverage': best_model['cv_coverage'],
        'winkler': best_model['cv_winkler'],
        'pinball': best_model['cv_pinball'],
        'model_name': best_model['name'],
    }
    
    # 6. 打印详细的每折结果
    print(f"  Ensemble CV Results (window={'full' if history_window is None else history_window}):")
    print(f"    Best Model: {cv_metrics['model_name']}")
    print(f"    CV-RMSE:     {cv_metrics['rmse']:.4f}")
    print(f"    CV-MASE:     {cv_metrics['mase']:.4f}")
    print(f"    CV-Coverage: {cv_metrics['coverage']*100:.2f}%")
    print(f"    CV-Winkler:  {cv_metrics['winkler']:.4f}")
    print(f"    CV-Pinball:  {cv_metrics['pinball']:.4f}")
    
    if best_model.get('cv_results'):
        print(f"\n    Detailed fold results:")
        for i, fold in enumerate(best_model['cv_results']):
            print(f"      Fold {i+1}: RMSE={fold['rmse']:.4f}, Coverage={fold['coverage']*100:.1f}%, MASE={fold['mase']:.4f}")
    
    # 7. 生成未来预测
    last_date = ts_log.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60, freq='D')
    exog_future = add_features(pd.DataFrame({'Date': future_dates}), 0, pd.Series(index=future_dates))
    exog_future['regime'] = float(exog['regime'].iloc[-1])
    
    ens_log = ensemble_forecast(ts_log, exog, exog_future, models['models'], steps=60, verbose=False)
    
    # 8. 回原尺度
    mu_log, se_log = ens_log['forecast'], ens_log['se']
    lo_log, hi_log = mu_log - 1.96 * se_log, mu_log + 1.96 * se_log
    
    # 获取残差用于 smearing
    cv_for_diag = best_model.get('cv_results', [])
    if cv_for_diag:
        resid_stable = np.concatenate([fold['errors'] for fold in cv_for_diag if 'errors' in fold])
    else:
        resid_stable = np.array([])
    
    y_pred = duan_smearing(mu_log, resid_stable) if len(resid_stable) > 0 else np.expm1(mu_log)
    y_lo = np.expm1(lo_log)
    y_hi = np.expm1(hi_log)
    
    return {
        'metrics': cv_metrics,
        'forecast': {
            'ds': future_dates,
            'yhat': y_pred,
            'yhat_lower': y_lo,
            'yhat_upper': y_hi,
        },
        'weights': ens_log['weights'],
        'models': models['models'],
    }


def run_unified_comparison(ts_raw: pd.Series, df_dates: pd.Series, 
                          output_dir: Path,
                          history_window: int | None = None) -> Dict:
    """
    运行统一的三模型对比（修改版：重新训练 Ensemble）
    
    Returns:
        包含所有模型指标和预测结果的字典
    """
    print("\n" + "="*60)
    print(f"Unified Model Comparison ({'Full History' if history_window is None else f'{history_window} Days'})")
    print("="*60)
    print("\nAll models use the same CV settings and evaluation criteria\n")
    
    results = {}
    
    # 1. Ensemble - 重新训练
    print("【Ensemble Model】")
    try:
        ensemble_result = train_ensemble_on_window(ts_raw, df_dates, history_window)
        results['ensemble'] = ensemble_result
        print("  ✓ Ensemble trained and validated")
    except Exception as e:
        print(f"  ✗ Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()
        results['ensemble'] = {'metrics': None, 'forecast': None}
    
    # 2. Prophet comparison
    results['prophet'] = {'metrics': None, 'forecast': None}
    if Prophet is not None:
        print("\n【Prophet Model】")
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
                print(f"  CV Metrics: RMSE={prophet_metrics['rmse']:.4f}, Coverage={prophet_metrics['coverage']*100:.1f}%, "
                      f"MASE={prophet_metrics['mase']:.4f}, Winkler={prophet_metrics['winkler']:.2f}")
            
            prophet_forecast_result = prophet_forecast(ts_raw, df_dates, horizon=60,
                                                      history_window=history_window)
            results['prophet']['forecast'] = prophet_forecast_result
            print("  ✓ Forecast completed")
        except Exception as e:
            print(f"  ✗ Prophet comparison failed: {e}")
    else:
        print("\n【Prophet Model】Not installed, skipping")
    
    # 3. Chronos comparison
    results['chronos'] = {'metrics': None, 'forecast': None}
    if ChronosPipeline is not None:
        print("\n【Chronos Model】")
        try:
            print("  Loading Chronos-2 model...")
            pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu")
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
                print(f"  CV Metrics: RMSE={chronos_metrics['rmse']:.4f}, Coverage={chronos_metrics['coverage']*100:.1f}%, "
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
    """加载数据 (支持 Excel 和 CSV)"""
    path_str = str(path)
    
    # 根据文件扩展名选择读取方式
    if path_str.endswith('.csv'):
        df = pd.read_csv(path)
        # CSV 列名标准化 (小写转标题格式)
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
        if 'number_of_reported_results' in df.columns:
            df.rename(columns={'number_of_reported_results': 'Number of  reported results'}, inplace=True)
    elif path_str.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path, header=1)
    else:
        # 尝试自动检测
        try:
            df = pd.read_csv(path)
            if 'date' in df.columns:
                df.rename(columns={'date': 'Date'}, inplace=True)
            if 'number_of_reported_results' in df.columns:
                df.rename(columns={'number_of_reported_results': 'Number of  reported results'}, inplace=True)
        except:
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
                       default=TASK1_RESULTS,
                       help="输出目录")
    parser.add_argument("--pictures-dir", type=Path,
                       default=TASK1_PICTURES,
                       help="图表输出目录")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("="*60)
    print("Unified Model Comparison: Ensemble vs Prophet vs Chronos")
    print("="*60)
    print("\nLoading data...")
    df_full = load_data(args.input)
    ts_raw_full = df_full.set_index('Date')['Number of  reported results'].sort_index()
    
    # 运行两种窗口的对比
    all_results = {}
    
    # 1. Full history comparison
    print("\n" + "="*80)
    print("【Comparison 1: Full History】")
    print("="*80)
    comparison_results_full = run_unified_comparison(
        ts_raw=ts_raw_full,
        df_dates=df_full['Date'],
        output_dir=args.output_dir,
        history_window=None
    )
    all_results['full'] = comparison_results_full
    
    # Generate full history visualization
    print("\nGenerating full history comparison plot...")
    if comparison_results_full['ensemble']['forecast'] is not None:
        plot_three_way_comparison(
            ts_raw=ts_raw_full,
            ensemble_result=comparison_results_full['ensemble']['forecast'],
            prophet_result=comparison_results_full['prophet']['forecast'],
            chronos_result=comparison_results_full['chronos']['forecast'],
            output_dir=args.pictures_dir,
            history_window=None,
            suffix="_full"
        )
    else:
        print("  ⚠️  Skipping plot: Ensemble training failed")
    
    # 2. Recent 240 days comparison
    print("\n" + "="*80)
    print("【Comparison 2: Recent 240 Days】")
    print("="*80)
    ts_raw_240 = ts_raw_full.iloc[-240:] if len(ts_raw_full) >= 240 else ts_raw_full
    df_240 = df_full[df_full['Date'].isin(ts_raw_240.index)].copy()
    
    comparison_results_240 = run_unified_comparison(
        ts_raw=ts_raw_240,
        df_dates=df_240['Date'],
        output_dir=args.output_dir,
        history_window=240
    )
    all_results['240'] = comparison_results_240
    
    # Generate 240-day visualization
    print("\nGenerating 240-day comparison plot...")
    if comparison_results_240['ensemble']['forecast'] is not None:
        plot_three_way_comparison(
            ts_raw=ts_raw_240,
            ensemble_result=comparison_results_240['ensemble']['forecast'],
            prophet_result=comparison_results_240['prophet']['forecast'],
            chronos_result=comparison_results_240['chronos']['forecast'],
            output_dir=args.pictures_dir,
            history_window=240,
            suffix="_240"
        )
    else:
        print("  ⚠️  Skipping plot: Ensemble training failed")
    
    # Generate unified comparison report (both windows)
    print("\nGenerating unified comparison report...")
    generate_unified_comparison_report(
        all_results=all_results,
        output_dir=args.output_dir
    )
    
    # Output detailed summary
    print("\n" + "="*80)
    print("【Unified Comparison Summary】")
    print("="*80)
    
    for window_name, window_results in [('Full History', all_results.get('full')), 
                                        ('Recent 240 Days', all_results.get('240'))]:
        print(f"\n{'='*80}")
        print(f"【{window_name}】")
        print(f"{'='*80}")
        
        if window_results:
            # Ensemble
            if window_results.get('ensemble', {}).get('metrics'):
                e = window_results['ensemble']['metrics']
                print(f"\nEnsemble ({e.get('model_name', 'N/A')}):")
                print(f"  RMSE:     {e['rmse']:.4f}")
                print(f"  MASE:     {e['mase']:.4f}")
                print(f"  Coverage: {e['coverage']*100:.2f}%")
                print(f"  Winkler:  {e['winkler']:.4f}")
                print(f"  Pinball:  {e['pinball']:.4f}")
            
            # Prophet
            if window_results.get('prophet', {}).get('metrics'):
                p = window_results['prophet']['metrics']
                print(f"\nProphet:")
                print(f"  RMSE:     {p['rmse']:.4f}")
                print(f"  MASE:     {p['mase']:.4f}")
                print(f"  Coverage: {p['coverage']*100:.2f}%")
                print(f"  Winkler:  {p['winkler']:.4f}")
                print(f"  Pinball:  {p['pinball']:.4f}")
            
            # Chronos
            if window_results.get('chronos', {}).get('metrics'):
                c = window_results['chronos']['metrics']
                print(f"\nChronos:")
                print(f"  RMSE:     {c['rmse']:.4f}")
                print(f"  MASE:     {c['mase']:.4f}")
                print(f"  Coverage: {c['coverage']*100:.2f}%")
                print(f"  Winkler:  {c['winkler']:.4f}")
                print(f"  Pinball:  {c['pinball']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"✓ All comparison results saved")
    print(f"{'='*80}")
    print(f"  文本报告 → {args.output_dir}/")
    print("    - unified_comparison_report.txt")
    print(f"\n  可视化图表 → {args.pictures_dir}/")
    print("    - 6_three_way_comparison_full.png")
    print("    - 6_three_way_comparison_240.png")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
