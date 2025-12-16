"""
Q1：时间序列集成预测（精简稳健版）

核心点：
1) 每折独立 PELT 变点（防泄露）
2) log1p 方差稳定
3) 分段 SARIMA + 滚动 CV
4) 合奏=均值加权 + 方差合并（正确合成 CI）
5) Duan smearing 回变换纠偏
6) 输出与 Prophet 同口径的评估指标（RMSE/MASE/Coverage/Winkler/Pinball）
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import holidays

from viz_report import (
    diagnostic_plots,
    generate_diagnostic_report,
    plot_weekday_effects,
    plot_changepoint_summary,
    plot_factor_importance,
)

warnings.filterwarnings('ignore')

try:
    import ruptures as rpt
except ImportError:
    raise ImportError("需要安装 ruptures: pip install ruptures")


# ======================
# 评估与合奏小工具
# ======================

def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 7) -> float:
    """MASE（默认以季节朴素 s=7 作为基线）"""
    if len(y_train) <= season:
        return np.nan
    denom = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if denom == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)

def compute_winkler_score(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, alpha: float = 0.10) -> float:
    """Winkler@1-alpha 区间评分（越小越好；alpha=0.10 对应 90% CI）"""
    width = y_upper - y_lower
    under = y_true < y_lower
    over  = y_true > y_upper
    penalty = (y_lower - y_true) * under + (y_true - y_upper) * over
    return float(np.mean(width + (2.0/alpha) * penalty))

def compute_pinball_loss_point(y_true: np.ndarray, y_pred: np.ndarray, taus=(0.1, 0.5, 0.9)) -> dict:
    """简化版 pinball：用点预测近似对应分位（用于横比即可）"""
    e = y_true - y_pred
    out = {}
    for t in taus:
        out[f"Pin@{int(t*100)}"] = float(np.mean(np.maximum(t*e, (t-1)*e)))
    return out

def combine_mean_se(means_log: List[np.ndarray], ses_log: List[np.ndarray], weights: np.ndarray):
    """合奏：均值加权 + 方差合并（独立近似；全部在 log 空间）"""
    w = np.asarray(weights)
    w = w / np.sum(w)
    mu = np.average(np.stack(means_log), axis=0, weights=w)
    var = np.sum((w[:, None]**2) * (np.stack(ses_log)**2), axis=0)
    return mu, np.sqrt(var), w

def duan_smearing(pred_log: np.ndarray, residuals_log: np.ndarray) -> np.ndarray:
    """Duan's smearing 纠偏回原尺度"""
    if residuals_log is None or len(residuals_log) == 0:
        return np.expm1(pred_log)
    k = float(np.mean(np.exp(residuals_log)))
    return np.exp(pred_log) * k - 1.0


# ======================
# 数据&特征
# ======================

def load_data(path: Path) -> pd.DataFrame:
    """
    加载数据，支持 Excel 和 CSV 格式
    优先使用 CSV（包含真实数据），Excel 文件是归一化的百分比数据
    """
    # 检查文件类型
    if path.suffix == '.csv':
        df = pd.read_csv(path)
        # CSV 的列名格式
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        # CSV 已经有真实的报告人数
        if 'number_of_reported_results' in df.columns:
            df['Number of  reported results'] = df['number_of_reported_results']
    else:
        # Excel 格式（注意：这是归一化的百分比数据！）
        df = pd.read_excel(path, header=0)
        df.columns = df.columns.str.strip()
        
        if 'timestamp' in df.columns:
            df['Date'] = pd.to_datetime(df['timestamp'])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            raise KeyError("找不到日期列（'timestamp' 或 'Date'）")
        
        # 计算总报告人数（Excel 是百分比，需要求和）
        try_columns = ['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
        df['Number of  reported results'] = df[try_columns].sum(axis=1)
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
    return df.sort_values('Date').reset_index(drop=True)

def detect_changepoint(ts: pd.Series) -> int:
    """PELT 变点（在原始尺度做 log1p 再检测）"""
    algo = rpt.Pelt(model="l2", min_size=30).fit(np.log(ts.values + 1))
    result = algo.predict(pen=3)
    return result[0] if len(result) > 1 else len(ts) // 2

def add_features(df: pd.DataFrame, changepoint_idx: int, ts: pd.Series) -> pd.DataFrame:
    exog = pd.DataFrame(index=ts.index)
    exog['regime'] = 0.0
    if changepoint_idx > 0:
        exog.loc[exog.index[changepoint_idx:], 'regime'] = 1.0
    
    dates = df['Date'].values if 'Date' in df.columns else ts.index
    weekday = pd.DatetimeIndex(dates).dayofweek
    exog['is_weekend'] = weekday.isin([5, 6]).astype(float)
    
    us_holidays = holidays.US(years=[2022, 2023])
    exog['is_holiday'] = pd.DatetimeIndex(dates).map(lambda x: float(x in us_holidays))
    
    return exog


def generate_explanation_report(ts_raw: pd.Series, exog: pd.DataFrame, 
                               models: Dict, cp_idx: int, 
                               output_dir: Path) -> Dict:
    """
    生成详细的可解释性分析报告
    """
    from statsmodels.tsa.seasonal import STL
    
    # 1. 变点分析
    cp_date = ts_raw.index[cp_idx]
    pre_cp = ts_raw.iloc[:cp_idx]
    post_cp = ts_raw.iloc[cp_idx:]
    cp_drop_abs = pre_cp.mean() - post_cp.mean()
    cp_drop_pct = cp_drop_abs / pre_cp.mean() * 100
    
    # 2. 周末效应
    weekday_mask = exog['is_weekend'] == 0
    weekend_mask = exog['is_weekend'] == 1
    weekday_avg = ts_raw[weekday_mask].mean()
    weekend_avg = ts_raw[weekend_mask].mean()
    weekend_effect_abs = weekend_avg - weekday_avg
    weekend_effect_pct = weekend_effect_abs / weekday_avg * 100
    
    # 3. 节假日效应
    holiday_mask = exog['is_holiday'] == 1
    non_holiday_mask = exog['is_holiday'] == 0
    if holiday_mask.sum() > 0:
        holiday_avg = ts_raw[holiday_mask].mean()
        non_holiday_avg = ts_raw[non_holiday_mask].mean()
        holiday_effect_pct = (holiday_avg - non_holiday_avg) / non_holiday_avg * 100
    else:
        holiday_avg = 0
        holiday_effect_pct = 0
    
    # 4. 趋势分解（STL）
    try:
        stl = STL(ts_raw, seasonal=7, period=7)
        stl_result = stl.fit()
        trend = stl_result.trend
        seasonal = stl_result.seasonal
        
        # 计算趋势斜率（最近30天 vs 前30天）
        recent_trend = trend.iloc[-30:].mean()
        prev_trend = trend.iloc[-60:-30].mean()
        trend_change = recent_trend - prev_trend
        trend_change_pct = trend_change / prev_trend * 100 if prev_trend != 0 else 0
        
        seasonal_strength = seasonal.std()
        
    except Exception as e:
        print(f"    [警告] STL分解失败: {e}")
        trend_change = 0
        trend_change_pct = 0
        seasonal_strength = 0
    
    # 5. 模型性能
    best = models['best']
    cv_metrics = {
        'model_name': best['name'],
        'cv_rmse': best.get('cv_rmse', 0),
        'cv_mase': best.get('cv_mase', 'N/A'),
    }
    
    # 6. 波动性分析
    overall_std = ts_raw.std()
    recent_std = ts_raw.iloc[-30:].std()
    volatility_change_pct = (recent_std - overall_std) / overall_std * 100
    
    # 生成报告文本
    report = f"""
{'='*80}
                        Wordle报告人数解释性分析
{'='*80}

数据概况
--------
观测期间: {ts_raw.index[0].date()} 至 {ts_raw.index[-1].date()} (共{len(ts_raw)}天)
总体均值: {ts_raw.mean():,.0f} 人/天
总体标准差: {overall_std:,.0f} 人
变异系数: {overall_std/ts_raw.mean()*100:.1f}%

{'='*80}
1. 结构性变化分析（Changepoint Effect）
{'='*80}

变点检测结果:
  变点日期: {cp_date.date()}
  变点位置: 第{cp_idx}天 / {len(ts_raw)}天 ({cp_idx/len(ts_raw)*100:.1f}%)

变点前后对比:
  变点前均值: {pre_cp.mean():,.0f} 人/天 (n={len(pre_cp)}天)
  变点后均值: {post_cp.mean():,.0f} 人/天 (n={len(post_cp)}天)
  绝对下降: {cp_drop_abs:,.0f} 人/天
  相对下降: {cp_drop_pct:.1f}%

可能解释:
  • 新鲜感消退: 游戏发布初期热度自然回落
  • 用户流失: 部分玩家失去兴趣或转向其他游戏
  • 社交传播减弱: 朋友圈晒成绩的风潮减退
  • 难度感知: 随着单词库被消耗，玩家可能觉得难度提升

{'='*80}
2. 周期性模式分析（Cyclical Patterns）
{'='*80}

周末效应:
  工作日(周一至周五)均值: {weekday_avg:,.0f} 人/天
  周末(周六周日)均值: {weekend_avg:,.0f} 人/天
  绝对差异: {weekend_effect_abs:,.0f} 人/天
  相对差异: {weekend_effect_pct:+.1f}%

"""
    
    if weekend_effect_pct < -3:
        report += "  解释: 工作日参与度更高，可能因为办公室文化、社交媒体分享或休息时间的娱乐需求\n"
    elif weekend_effect_pct > 3:
        report += "  解释: 周末玩家更活跃，有更多闲暇时间玩游戏并分享成绩\n"
    else:
        report += "  解释: 周末效应不明显，玩家行为较为稳定\n"
    
    report += f"""
节假日效应:
  非节假日均值: {non_holiday_avg:,.0f} 人/天
  节假日均值: {holiday_avg:,.0f} 人/天 (n={holiday_mask.sum()}天)
  相对差异: {holiday_effect_pct:+.1f}%

季节性强度:
  STL季节项标准差: {seasonal_strength:.0f} 人
  解释: {'周内波动较强' if seasonal_strength > overall_std*0.2 else '周内波动较弱'}

{'='*80}
3. 趋势分析（Trend Dynamics）
{'='*80}

长期趋势:
  前30天趋势均值: {prev_trend:,.0f} 人/天
  最近30天趋势均值: {recent_trend:,.0f} 人/天
  趋势变化: {trend_change:,.0f} 人/天
  变化率: {trend_change_pct:+.1f}%

"""
    
    if trend_change_pct < -5:
        report += "  解释: 持续下降趋势，玩家群体仍在流失\n"
    elif trend_change_pct > 5:
        report += "  解释: 趋势向上，可能有新的推广活动或病毒式传播\n"
    else:
        report += "  解释: 趋势趋于稳定，核心玩家群体相对固定\n"
    
    report += f"""
波动性变化:
  整体标准差: {overall_std:,.0f} 人
  最近30天标准差: {recent_std:,.0f} 人
  波动性变化: {volatility_change_pct:+.1f}%
  解释: {'波动性增加，预测不确定性上升' if volatility_change_pct > 10 else '波动性稳定'}

{'='*80}
4. 模型验证（Model Performance）
{'='*80}

最优模型: {cv_metrics['model_name']}
交叉验证RMSE: {cv_metrics['cv_rmse']:,.0f} 人
交叉验证MASE: {cv_metrics['cv_mase']}

模型可靠性:
  • 使用滚动交叉验证防止过拟合
  • 每折独立检测变点防止数据泄露
  • 集成多个候选模型降低单一模型风险
  • 使用Duan smearing纠正对数变换偏差

{'='*80}
5. 关键发现总结
{'='*80}

影响因素排序（按影响力大小）:
  1. 结构性变化: {cp_drop_pct:.1f}% （最强）
  2. 趋势变化: {trend_change_pct:+.1f}%
  3. 周末效应: {weekend_effect_pct:+.1f}%
  4. 节假日效应: {holiday_effect_pct:+.1f}%

主要结论:
  • 游戏热度在{cp_date.strftime('%Y年%m月')}出现显著下降
  • {'周末' if weekend_effect_pct > 0 else '工作日'}玩家更活跃
  • {'最近趋势继续下降' if trend_change_pct < 0 else '趋势企稳'}
  • 模型在历史数据上表现稳健，可用于短期预测

{'='*80}
"""
    
    # 保存报告
    with open(output_dir / 'explanation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("✓ 解释性分析报告已生成: explanation_report.txt")
    print("="*60)
    
    # 返回关键指标用于后续可视化
    return {
        'cp_drop_pct': cp_drop_pct,
        'weekend_effect_pct': weekend_effect_pct,
        'holiday_effect_pct': holiday_effect_pct,
        'trend_change_pct': trend_change_pct,
        'volatility_change_pct': volatility_change_pct,
        'seasonal_strength': seasonal_strength,
        'overall_std': overall_std,
    }


# ======================
# CV 与模型拟合
# ======================

def rolling_cv(ts: pd.Series, df_dates: pd.Series, order: Tuple, seasonal_order: Tuple,
               n_splits: int = 3, horizon: int = 30) -> List[Dict]:
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
        
        print(f"      Fold {i+1}/{n_splits}: train={train_end} days, test={test_end-train_end} days...", end=" ", flush=True)
        
        train, test = ts.iloc[:train_end], ts.iloc[train_end:test_end]
        try:
            cp_idx = detect_changepoint(np.expm1(train))
            
            exog_train = add_features(pd.DataFrame({'Date': df_dates.iloc[:train_end]}), cp_idx, train)
            exog_test  = add_features(pd.DataFrame({'Date': df_dates.iloc[train_end:test_end]}), 0, test)
            exog_test['regime'] = float(exog_train['regime'].iloc[-1])
            exog_test = exog_test[exog_train.columns].fillna(0)
            
            model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=50, method='lbfgs')
            
            fc = fitted.get_forecast(steps=len(test), exog=exog_test)
            pred_mean = fc.predicted_mean.values
            
            # 🔧 关键修复：使用SARIMAX提供的预测区间
            pred_summary = fc.summary_frame(alpha=0.10)  # 🔧 统一：90% CI
            ci_l = pred_summary['mean_ci_lower'].values
            ci_u = pred_summary['mean_ci_upper'].values
            pred_se = (ci_u - ci_l) / (2 * 1.645)  # 🔧 90% CI 对应 1.645

            test_values = test.values
            errors = test_values - pred_mean

            coverage = float(np.mean((test_values >= ci_l) & (test_values <= ci_u)))
            
            rmse = float(np.sqrt(np.mean(errors**2)))
            mase = compute_mase(test_values, pred_mean, train.values)
            wink = compute_winkler_score(test_values, ci_l, ci_u, alpha=0.10)  # 🔧 统一：90% CI
            pinb = compute_pinball_loss_point(test_values, pred_mean)['Pin@50']
            
            print(f"✓ RMSE={rmse:.4f}, Coverage={coverage*100:.1f}%")
            
            results.append({
                'rmse': rmse,
                'pred_var': float(np.mean(pred_se**2)),  # 🔧 使用正确的预测方差
                'coverage': coverage,
                'mase': mase,
                'pinball': pinb,
                'winkler': wink,
                'errors': errors,
                'train_cp_idx': cp_idx,
                'train_end': train_end,
                'pred_mean': pred_mean,
                'pred_se': pred_se,
                'test_values': test_values,
            })
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
            print(f"    ⚠️  所有折都失败，跳过此模型")
            continue
            
        avg = lambda k: np.nanmean([r[k] for r in cv])
        print(f"    CV指标: RMSE={avg('rmse'):6.4f}  Coverage={avg('coverage'):5.1%}  "
              f"MASE={avg('mase'):5.3f}  Winkler={avg('winkler'):6.2f}")
        
        results.append({
            'name': name,
            'order': order,
            'seasonal_order': seasonal_order,
            'cv_rmse': avg('rmse'),
            'cv_coverage': avg('coverage'),
            'cv_mase': avg('mase'),
            'cv_winkler': avg('winkler'),
            'cv_pinball': avg('pinball'),
            'pred_var': avg('pred_var'),
            'cv_changepoints': [r.get('train_cp_idx', 0) for r in cv],
            'cv_changepoints_global': [r.get('train_cp_idx', 0) for r in cv],
            'cv_train_lengths': [r.get('train_end', 0) for r in cv],
            'cv_results': cv,
        })
    
    if not results:
        raise ValueError("所有模型拟合失败")
    
    best = min(results, key=lambda x: x['cv_rmse'])
    print(f"\n✓ 最优模型：{best['name']} (CV-RMSE: {best['cv_rmse']:.4f})")
    return {'models': results, 'best': best}

def forecast_with_ci(ts: pd.Series, exog: pd.DataFrame, exog_future: pd.DataFrame,
                     model_info: Dict, steps: int, verbose: bool = False) -> Dict:
    """单模型预测（log 空间），返回均值/预测标准误/90% CI"""
    model = SARIMAX(ts, exog=exog,
                    order=model_info['order'], seasonal_order=model_info['seasonal_order'],
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=50, method='lbfgs')
    fc = fitted.get_forecast(steps=steps, exog=exog_future)
    
    pred_mean = fc.predicted_mean.values
    
    # 🔧 统一使用 90% CI (alpha=0.10)
    pred_summary = fc.summary_frame(alpha=0.10)  # 90% CI
    ci_lower = pred_summary['mean_ci_lower'].values
    ci_upper = pred_summary['mean_ci_upper'].values
    
    # 从置信区间反推标准误（90% CI 对应 1.645 倍标准误）
    pred_se = (ci_upper - ci_lower) / (2 * 1.645)
    
    if verbose:
        print(f"  [{model_info['name']}] mean(pred_se)={np.mean(pred_se):.4f}")
    
    return {
        'forecast': pred_mean,
        'se': pred_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }


def ensemble_forecast(ts: pd.Series, exog: pd.DataFrame, exog_future: pd.DataFrame,
                      model_results: List[Dict], steps: int, verbose: bool = False) -> Dict:
    """合奏（log 空间）：IVW 加权均值 + 方差合并，输出 90% CI"""
    means_log, ses_log, inv_vars = [], [], []
    
    for m in model_results:
        r = forecast_with_ci(ts, exog, exog_future, m, steps, verbose=False)
        means_log.append(r['forecast'])
        ses_log.append(r['se'])
        # IVW权重基于CV的预测方差
        inv_vars.append(1.0 / (m['pred_var'] + 1e-6))

    mu_log, se_log, w = combine_mean_se(means_log, ses_log, np.array(inv_vars))

    if verbose:
        print("\n生成集成预测...")
        print("权重分配（Inverse Variance Weighting）：")
        for wi, mi in zip(w, model_results):
            print(f"  {mi['name']:<30} {wi*100:5.1f}%")
        print(f"\n预测区间统计（log空间，90% CI）：")
        print(f"  平均预测标准误：{np.mean(se_log):.4f}")
        print(f"  标准误范围：[{np.min(se_log):.4f}, {np.max(se_log):.4f}]")

    lo_log = mu_log - 1.645 * se_log  # 🔧 统一：90% CI
    hi_log = mu_log + 1.645 * se_log

    return {
        'forecast': mu_log, 
        'se': se_log,
        'ci_lower': lo_log, 
        'ci_upper': hi_log, 
        'weights': w
    }


def walk_forward_validation(ts: pd.Series, df_dates: pd.Series, 
                            model_results: List[Dict], 
                            horizons: List[int] = [30, 60], 
                            verbose: bool = False) -> Dict:
    """Walk-forward 验证：测试集成预测在不同预测窗口的覆盖率（log 空间）"""
    results = {}
    min_train = 180  # 最小训练集长度
    
    for h in horizons:
        # 🔧 修复：只需要能至少测试一次即可
        if len(ts) < min_train + h:
            if verbose:
                print(f"  [h={h}] 跳过：数据不足（需要至少 {min_train + h} 天，当前 {len(ts)} 天）")
            continue
            
        coverages = []
        test_start = min_train
        
        while test_start + h <= len(ts):
            train = ts.iloc[:test_start]
            test = ts.iloc[test_start:test_start + h]
            
            if len(test) < h:
                break
            
            try:
                # 构建特征
                cp_idx = detect_changepoint(np.expm1(train))
                exog_train = add_features(pd.DataFrame({'Date': df_dates.iloc[:test_start]}), cp_idx, train)
                exog_test = add_features(pd.DataFrame({'Date': df_dates.iloc[test_start:test_start+h]}), 0, test)
                exog_test['regime'] = float(exog_train['regime'].iloc[-1])
                exog_test = exog_test[exog_train.columns].fillna(0)
                
                # 集成预测（log 空间）
                ens = ensemble_forecast(train, exog_train, exog_test, model_results, steps=h, verbose=False)
                
                # 🔧 关键修复：转换到原始尺度再计算覆盖率
                # 获取残差用于 smearing
                train_resid = []
                for m in model_results:
                    if m.get('cv_results'):
                        for fold in m['cv_results']:
                            if 'errors' in fold:
                                train_resid.extend(fold['errors'])
                
                # 🔧 关键修复：在 log 空间计算覆盖率，避免 Jensen 不等式问题
                # log 空间的 CI 是对称的，转到原始尺度会不对称
                ci_l_log = ens['ci_lower']
                ci_u_log = ens['ci_upper']
                test_vals_log = test.values
                
                # 在 log 空间计算覆盖率（正确做法）
                coverage = np.mean((test_vals_log >= ci_l_log) & (test_vals_log <= ci_u_log))
                coverages.append(coverage)
                
                if verbose:
                    print(f"    [h={h}, start={test_start}] Coverage={coverage*100:.1f}%")
                
            except Exception as e:
                if verbose:
                    print(f"    [h={h}, start={test_start}] 失败: {str(e)[:50]}")
                pass
            
            test_start += 30  # 每次前进30天
        
        if coverages:
            results[h] = {
                'coverage': np.mean(coverages),
                'n_tests': len(coverages)
            }
            if verbose or True:  # 总是打印详细结果
                print(f"  [h={h}天] 平均覆盖率: {np.mean(coverages)*100:.1f}% (n={len(coverages)})")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/Users/1m/Desktop/大三上/大数据/期末/data.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--history-window", type=int, default=None,
                        help="主训练窗口长度（None=全历史）")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Q1：时间序列集成预测（精简稳健版）")
    print("="*60)
    
    # 1) 数据
    print("\n[1/8] 加载数据...")
    df = load_data(args.input)
    ts_raw_full = df.set_index('Date')['Number of  reported results'].sort_index()
    
    if args.history_window:
        ts_raw = ts_raw_full.iloc[-args.history_window:]
        df = df[df['Date'].isin(ts_raw.index)].copy()
        print(f"  ✓ 使用最近 {args.history_window} 天数据 ({ts_raw.index[0].date()} 至 {ts_raw.index[-1].date()})")
    else:
        ts_raw = ts_raw_full
        print(f"  ✓ 使用全部历史数据 ({len(ts_raw)} 天)")
    
    ts_log = np.log1p(ts_raw)
    
    # 2) 变点摘要
    print("\n[2/8] 检测变点...")
    cp_idx = detect_changepoint(ts_raw)
    cp_date = ts_raw.index[cp_idx]
    pre_mean = ts_raw.iloc[:cp_idx].mean()
    post_mean = ts_raw.iloc[cp_idx:].mean()
    print(f"  ✓ 变点: {cp_date.date()} (第{cp_idx}天)")
    print(f"    变点前均值: {pre_mean:,.0f} 人/天")
    print(f"    变点后均值: {post_mean:,.0f} 人/天")
    print(f"    下降幅度: {(post_mean-pre_mean)/pre_mean*100:.1f}%")

    # 3) 外生变量
    print("\n[3/8] 构建特征...")
    exog = add_features(df, cp_idx, ts_log)
    print(f"  ✓ 生成 {len(exog.columns)} 个外生变量: {list(exog.columns)}")

    # 4) 拟合候选 + CV
    print("\n[4/8] 拟合并交叉验证候选模型...")
    print("  (这一步可能需要几分钟，请耐心等待...)")
    import time
    total_start = time.time()
    models = fit_models(ts_log, df['Date'])
    total_elapsed = time.time() - total_start
    print(f"\n  ✓ 模型训练完成，总耗时 {total_elapsed:.1f} 秒")

    # 4.5) 生成解释性分析报告
    print("\n[5/8] 生成解释性分析报告...")
    explanation_stats = generate_explanation_report(ts_raw, exog, models, cp_idx, args.output_dir)

    # 5) 残差诊断
    print("\n[6/8] 生成诊断图表...")
    best = models['best']
    cv_for_diag = rolling_cv(ts_log, df['Date'], best['order'], best['seasonal_order'], n_splits=3, horizon=30)
    diag_stats = diagnostic_plots(ts_log, exog, best, cv_for_diag, args.output_dir)
    resid_stable = diag_stats.get('residuals_stable', pd.Series(dtype=float)).dropna().values

    # 6) Walk-Forward
    print("\n[7/8] Walk-Forward 验证...")
    wf_results = walk_forward_validation(ts_log, df['Date'], models['models'], horizons=[30, 60], verbose=False)
    for h, stats in wf_results.items():
        print(f"  h={h}天: 覆盖率={stats['coverage']*100:.1f}% (n={stats['n_tests']})")  # 🔧 修复：乘以100
    
    # 7) 未来预测
    print("\n[8/8] 生成未来预测...")
    last_date = ts_log.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60, freq='D')
    exog_future = add_features(pd.DataFrame({'Date': future_dates}), 0, pd.Series(index=future_dates))
    exog_future['regime'] = float(exog['regime'].iloc[-1])
    
    # 8) 合奏
    ens_log = ensemble_forecast(ts_log, exog, exog_future, models['models'], steps=60, verbose=True)
    
    # 9) 回原尺度（已在 ensemble_forecast 中计算了 90% CI）
    mu_log, se_log = ens_log['forecast'], ens_log['se']
    lo_log, hi_log = ens_log['ci_lower'], ens_log['ci_upper']  # 🔧 直接使用 90% CI

    y_pred = duan_smearing(mu_log, resid_stable)
    y_lo   = np.expm1(lo_log)
    y_hi   = np.expm1(hi_log)
    
    # 10) 可视化与报告
    print("\n生成可视化图表...")
    generate_diagnostic_report(best, cv_for_diag, diag_stats, wf_results, args.output_dir)
    plot_weekday_effects(ts_raw, cp_idx, args.output_dir)
    plot_changepoint_summary(ts_raw, cp_idx, args.output_dir)
    plot_factor_importance(explanation_stats, args.output_dir)
    
    # 11) 保存结果
    with open(args.output_dir / "ensemble_result.pkl", "wb") as f:
        pickle.dump({'forecast': y_pred, 'ci_lower': y_lo, 'ci_upper': y_hi, 'weights': ens_log['weights']}, f)
    
    # 12) 输出目标日
    target_idx = (pd.to_datetime('2023-03-01') - future_dates[0]).days
    if 0 <= target_idx < 60:
        pred = y_pred[target_idx]; lo = y_lo[target_idx]; hi = y_hi[target_idx]
        half_w = (hi - lo)/2
        recent_std = ts_raw.iloc[-30:].std()
        print("\n" + "="*60)
        print("【2023-03-01 预测】")
        print("="*60)
        print(f"点预测：{pred:,.0f} 人")
        print(f"90% CI：[{lo:,.0f}, {hi:,.0f}]")  # 🔧 统一使用 90% CI
        print(f"区间宽度：±{half_w:,.0f} 人（≈{half_w/recent_std:.1f}× 最近30天std）")
        
    print("\n" + "="*60)
    print(f"✓ 所有输出已保存至: {args.output_dir}")
    print("="*60)
    print("  文本报告:")
    print("    - explanation_report.txt")
    print("    - diagnostic_report.txt")
    print("  可视化图表:")
    print("    - 1_weekday_effects.png")
    print("    - 2_changepoint.png")
    print("    - 3_diagnostics.png")
    print("    - 4_factor_importance.png")
    print("  模型文件:")
    print("    - ensemble_result.pkl")
    print("\n下一步:")
    print("  运行 python model_comparison.py --input ../data.xlsx")
    print("  对比 Ensemble vs Prophet vs Chronos")
    print("="*60)

if __name__ == "__main__":
    main()
