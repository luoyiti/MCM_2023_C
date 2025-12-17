"""Visualization and reporting helpers with narrative-focused plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

COLORS = {
    "primary": "#2E4057",    # 深海军蓝 - 稳重专业
    "secondary": "#8B4789",  # 深紫罗兰 - 高雅神秘
    "accent": "#C97064",     # 赤陶橙 - 温暖有力
    "success": "#3A7D7D",    # 深青绿 - 沉稳可靠
    "warning": "#D4A373",    # 古铜金 - 典雅高贵
    "dark": "#2F2F2F",       # 炭黑色 - 经典稳重
    "light": "#F5F5F5",      # 珍珠白 - 纯净柔和
}

# 莫兰迪色系 - 统一因素配色方案
MORANDI_COLORS = {
    'Structural Change\n(Changepoint)': '#A8917D',  # 莫兰迪棕 - 最重要的因素用沉稳色
    'Changepoint': '#A8917D',  # 同一因素别名
    'Weekend Effect': '#C9A7A6',  # 莫兰迪粉 - 柔和温暖
    'Holiday Effect': '#9EB5B0',  # 莫兰迪绿 - 清新淡雅
    'Trend Change\n(Last 30d)': '#A5B5C4',  # 莫兰迪蓝 - 理性冷静
    'Trend Change': '#A5B5C4',  # 同一因素别名
    'Volatility Change': '#B8A7B2',  # 莫兰迪紫 - 神秘优雅
    'Current Word\n(lag0)': '#E8C4A3',  # 莫兰迪浅橙 - 代表当天单词属性
    'Previous Word\n(lag1)': '#D4B896',  # 莫兰迪米黄 - 代表前一天单词属性
    'Word Attributes\n(Lagged Features)': '#D4B896',  # 兼容旧版本
    'Word Attributes': '#D4B896',  # 同一因素别名
}


# --- Diagnostics & report -------------------------------------------------
def diagnostic_plots(ts, exog, model_info: Dict, cv_results: List[Dict], output_dir: Path):
    """Residual diagnostics on stable regime."""
    model = SARIMAX(
        ts,
        exog=exog,
        order=model_info["order"],
        seasonal_order=model_info["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=50)
    residuals = fitted.resid

    regime_change_idx = (
        np.where(exog["regime"].values == 1)[0][0] if (exog["regime"] == 1).any() else 0
    )
    residuals_stable = residuals.iloc[regime_change_idx:]

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("white")

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(residuals_stable.index, residuals_stable.values, color=COLORS["primary"], linewidth=1.2)
    ax1.axhline(0, color=COLORS["dark"], linestyle="--", linewidth=1.2, alpha=0.6)
    std = residuals_stable.std()
    ax1.fill_between(residuals_stable.index, -2 * std, 2 * std, color=COLORS["light"], alpha=0.4)
    lb_test = acorr_ljungbox(residuals_stable.dropna(), lags=[10, 20], return_df=True)
    p_value = lb_test["lb_pvalue"].iloc[-1]
    ax1.set_title("Residuals (stable regime)", fontsize=12, fontweight="bold")
    ax1.text(
        0.02,
        0.96,
        f"Ljung-Box p={p_value:.4f}\nn={len(residuals_stable)}",
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )
    ax1.grid(True, alpha=0.3, linestyle=":")

    ax2 = plt.subplot(2, 2, 2)
    plot_acf(
        residuals_stable.dropna(),
        lags=min(30, len(residuals_stable) // 2),
        ax=ax2,
        color=COLORS["primary"],
        alpha=0.6,
        zero=False,
    )
    ax2.set_title("ACF", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle=":")

    ax3 = plt.subplot(2, 2, 3)
    resid_clean = residuals_stable.dropna()
    ax3.hist(resid_clean, bins=25, density=True, color=COLORS["accent"], alpha=0.6, edgecolor="white")
    mu, sigma = resid_clean.mean(), resid_clean.std()
    x = np.linspace(resid_clean.min(), resid_clean.max(), 100)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS["warning"], linewidth=2.2, label=f"N({mu:.3f},{sigma:.3f}²)")
    _, sw_p = stats.shapiro(resid_clean)
    ax3.text(
        0.02,
        0.96,
        f"Shapiro p={sw_p:.4f}",
        transform=ax3.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax3.set_title("Residual distribution", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle=":")

    ax4 = plt.subplot(2, 2, 4)
    max_horizon = max(len(r["errors"]) for r in cv_results)
    coverage_by_h = []
    for h in range(max_horizon):
        vals = []
        for cv in cv_results:
            if len(cv["errors"]) > h:
                std_h = np.std([r["errors"][h] for r in cv_results if len(r["errors"]) > h])
                vals.append(abs(cv["errors"][h]) <= 1.96 * std_h)
        coverage_by_h.append(np.mean(vals) * 100 if vals else np.nan)
    horizons = np.arange(1, len(coverage_by_h) + 1)
    ax4.plot(horizons, coverage_by_h, color=COLORS["success"], marker="o", linewidth=2)
    ax4.axhline(95, color=COLORS["warning"], linestyle="--", alpha=0.7)
    ax4.fill_between(horizons, 90, 100, color="lightgreen", alpha=0.2)
    ax4.set_ylim(80, 105)
    ax4.set_title("Empirical coverage vs horizon", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_dir / "3_diagnostics.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return {
        "n_residuals": len(residuals_stable),
        "mean": mu,
        "std": sigma,
        "skewness": stats.skew(resid_clean),
        "kurtosis": stats.kurtosis(resid_clean),
        "ljung_box_p": p_value,
        "shapiro_p": sw_p,
        "residuals_stable": residuals_stable,
    }


def generate_diagnostic_report(model_info: Dict, cv_results: List[Dict], diag_stats: Dict, wf_results: Dict, output_dir: Path, prophet_metrics: Dict | None = None):
    lines = [
        "=" * 70,
        "模型诊断报告",
        "=" * 70,
        f"\n模型: {model_info['name']}",
        f"CV-RMSE: {model_info['cv_rmse']:.4f} (log空间)",
        f"CV覆盖率: {model_info['cv_coverage']:.1%} (理想: 95%)",
        f"\n残差诊断 (n={diag_stats['n_residuals']}):",
        f"  均值={diag_stats['mean']:.4f}, 标准差={diag_stats['std']:.4f}",
        f"  Ljung-Box p={diag_stats['ljung_box_p']:.4f}, Shapiro-Wilk p={diag_stats['shapiro_p']:.4f}",
    ]
    
    # Walk-Forward验证结果（如果有）
    if wf_results:
        avg_wf_cov = np.mean([r["coverage"] for r in wf_results.values()])
        h60_cov = wf_results.get(60, {}).get("coverage", 0)
        lines.append("\nWalk-Forward验证 (集成区间覆盖率，90% CI):")
        for h in sorted(wf_results.keys()):
            res = wf_results[h]
            marker = "⭐" if h == 60 else "  "
            lines.append(f"{marker} h={h:2d}天: {res['coverage']*100:5.1f}% (n={res['n_tests']})")  # 🔧 乘以100
        lines.extend([
            f"\n平均覆盖率: {avg_wf_cov*100:.1f}% (目标≈90%)",  # 🔧 乘以100
            f"h=60覆盖率: {h60_cov*100:.1f}% (2023-03-01预测)",  # 🔧 乘以100
        ])
    
    # CV变点检测信息（显示相对索引和训练集长度）
    cp_info = model_info.get('cv_changepoints', [])
    train_lengths = model_info.get('cv_train_lengths', [])
    if cp_info and train_lengths:
        cp_str = ", ".join([f"折{i}: 位置{cp} (训练集长度={tl})" 
                           for i, (cp, tl) in enumerate(zip(cp_info, train_lengths))])
        lines.append(f"\nCV变点检测 (避免泄露): {cp_str}")
        # 如果所有折检测到相同位置，添加说明
        if len(set(cp_info)) == 1 and len(cp_info) > 1:
            lines.append(f"  注意：所有折检测到相同变点位置（相对索引={cp_info[0]}），")
            lines.append(f"        可能因为数据在早期有明显变点，且所有折的训练集都从序列开头开始")
    else:
        lines.append(f"\nCV变点检测 (避免泄露): {cp_info}")
    
    lines.extend([
        "\n对比模型：Prophet" if prophet_metrics else "",
        f"  CV-RMSE={prophet_metrics['rmse']:.4f}, Coverage={prophet_metrics['coverage']:.3f}" if prophet_metrics else "",
        "=" * 70,
    ])
    
    report = "\n".join(lines)
    (output_dir / "diagnostic_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


# --- Storytelling visuals -------------------------------------------------
def plot_weekday_effects(ts_raw: pd.Series, changepoint_idx: int, output_dir: Path):
    """Bar chart of weekday effects pre/post changepoint (clean alternative to heatmap)."""
    df = ts_raw.reset_index()
    df["dow"] = df["Date"].dt.day_name().str[:3]
    cp_date = ts_raw.index[changepoint_idx]
    df["phase"] = np.where(df["Date"] <= cp_date, "Pre-shift", "Post-shift")

    agg = df.groupby(["phase", "dow"])["Number of  reported results"].mean().reset_index()
    pivot = agg.pivot(index="dow", columns="phase", values="Number of  reported results")
    pivot = pivot.loc[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]  # order

    x = np.arange(len(pivot.index))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, pivot["Pre-shift"], width, color=COLORS["primary"], alpha=0.7, label="Pre-shift")
    ax.bar(x + width / 2, pivot["Post-shift"], width, color=COLORS["secondary"], alpha=0.7, label="Post-shift")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Mean reported results")
    ax.set_title("Weekday effects (Pre vs Post shift)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_dir / "1_weekday_effects.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_changepoint_summary(ts_raw: pd.Series, changepoint_idx: int, output_dir: Path):
    """Simple changepoint view with two mean bands."""
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("white")
    ax.plot(ts_raw.index, ts_raw.values, color=COLORS["dark"], linewidth=2, alpha=0.8)
    cp_date = ts_raw.index[changepoint_idx]
    ax.axvline(cp_date, color=COLORS["warning"], linestyle="--", linewidth=2.2, label="PELT changepoint")

    mean_before = ts_raw.iloc[:changepoint_idx].mean()
    mean_after = ts_raw.iloc[changepoint_idx:].mean()
    ax.fill_between(
        ts_raw.index[:changepoint_idx],
        mean_before * 0.95,
        mean_before * 1.05,
        color=COLORS["primary"],
        alpha=0.25,
        label="Phase mean (before)",
    )
    ax.fill_between(
        ts_raw.index[changepoint_idx:],
        mean_after * 0.95,
        mean_after * 1.05,
        color=COLORS["success"],
        alpha=0.25,
        label="Phase mean (after)",
    )
    ax.text(
        cp_date,
        ts_raw.max() * 0.9,
        "Shift into stable play period",
        fontsize=10,
        color=COLORS["warning"],
        ha="left",
        va="center",
    )
    ax.set_title("Changepoint and Phase Means", fontsize=14, fontweight="bold")


def plot_factor_importance(explanation_stats: Dict, output_dir: Path):
    """
    可视化各影响因素的相对重要性（改进版：包含滞后特征）
    用于MCM论文的解释性分析部分
    使用莫兰迪色系，确保同一因素在不同图表中颜色一致
    """
    # 提取各因素的百分比影响
    factors = {
        'Structural Change\n(Changepoint)': abs(explanation_stats.get('cp_drop_pct', 0)),
        'Weekend Effect': abs(explanation_stats.get('weekend_effect_pct', 0)),
        'Holiday Effect': abs(explanation_stats.get('holiday_effect_pct', 0)),
        'Trend Change\n(Last 30d)': abs(explanation_stats.get('trend_change_pct', 0)),
        'Volatility Change': abs(explanation_stats.get('volatility_change_pct', 0)),
    }
    
    # 添加当天单词属性的影响（如果存在）
    lag0_impact = explanation_stats.get('lag0_features_impact', None)
    if lag0_impact and lag0_impact > 0:
        factors['Current Word\n(lag0)'] = abs(lag0_impact)
    
    # 添加滞后单词属性的影响（如果存在）
    lag1_impact = explanation_stats.get('lag1_features_impact', None)
    if lag1_impact and lag1_impact > 0:
        factors['Previous Word\n(lag1)'] = abs(lag1_impact)
    
    # 按重要性排序
    sorted_factors = dict(sorted(factors.items(), key=lambda x: x[1], reverse=True))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # 子图1：横向柱状图（展示绝对值）
    # 🎨 使用统一的莫兰迪配色方案
    factor_names = list(sorted_factors.keys())
    colors_list = [MORANDI_COLORS.get(name, '#B5B5B5') for name in factor_names]
    
    y_pos = np.arange(len(sorted_factors))
    values = list(sorted_factors.values())
    
    bars = ax1.barh(y_pos, values, color=colors_list, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(factor_names, fontsize=11)
    ax1.set_xlabel('Impact Magnitude (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Factor Importance Ranking', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(val + max(values)*0.02, i, f'{val:.1f}%', 
                va='center', fontsize=10, fontweight='bold', color='#333333')
    
    # 子图2：饼图（展示相对占比）
    total = sum(values)
    percentages = [v/total*100 for v in values]
    
    # 🎨 使用相同的莫兰迪配色，确保两张图颜色一致
    # 创建简洁标签供图例使用
    labels_for_legend = []
    for name in factor_names:
        if '\n' in name:
            # 移除换行符，保持简洁
            labels_for_legend.append(name.replace('\n(Changepoint)', '').replace('\n(Last 30d)', ''))
        else:
            labels_for_legend.append(name)
    
    # 创建饼图，使用autopct显示百分比，pctdistance控制百分比位置
    wedges, texts, autotexts = ax2.pie(
        percentages, 
        labels=None,  # 先不显示标签
        autopct='%1.1f%%',
        startangle=45,  # 调整起始角度避免重叠
        colors=colors_list,  # 🎨 使用与柱状图相同的莫兰迪配色
        textprops={'fontsize': 10},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},  # 加宽白边，更清晰
        pctdistance=0.75  # 百分比文本距离中心的距离
    )
    
    # 增强百分比文本 - 使用深灰色而非白色，更柔和
    for autotext in autotexts:
        autotext.set_color('#FFFFFF')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # 创建图例（放在饼图外侧，避免重叠）
    ax2.legend(
        labels_for_legend,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,  # 去掉阴影，更简洁
        edgecolor='#DDDDDD'
    )
    
    ax2.set_title('Relative Contribution', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_factor_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 因素重要性可视化已保存: 4_factor_importance.png")


def plot_changepoint_details(ts_raw: pd.Series, changepoint_idx: int, output_dir: Path):
    """Complete changepoint visualization (继续原有函数)"""
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("white")
    cp_date = ts_raw.index[changepoint_idx]
    ax.plot(ts_raw.index, ts_raw.values, color=COLORS["dark"], linewidth=2, alpha=0.8)
    ax.axvline(cp_date, color=COLORS["warning"], linestyle="--", linewidth=2.2, label="PELT changepoint")
    
    mean_before = ts_raw.iloc[:changepoint_idx].mean()
    mean_after = ts_raw.iloc[changepoint_idx:].mean()
    ax.fill_between(
        ts_raw.index[:changepoint_idx],
        mean_before * 0.95,
        mean_before * 1.05,
        color=COLORS["primary"],
        alpha=0.25,
        label="Phase mean (before)",
    )
    ax.fill_between(
        ts_raw.index[changepoint_idx:],
        mean_after * 0.95,
        mean_after * 1.05,
        color=COLORS["success"],
        alpha=0.25,
        label="Phase mean (after)",
    )
    
    ax.set_title("Changepoint and Phase Means", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reported results")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_dir / "2_changepoint.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_fan_calibration(ts_raw: pd.Series, base_result: Dict, calibrated_result: Dict, output_dir: Path):
    """Fan chart with pre/post calibration 90% bands and target annotation."""
    forecast_dates = pd.date_range(
        start=ts_raw.index[-1] + pd.Timedelta(days=1), periods=len(calibrated_result["forecast"]), freq="D"
    )
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    # focus on recent 60 days to reduce scale compression
    window_days = 60
    hist_start = max(0, len(ts_raw) - window_days)
    ax.plot(
        ts_raw.index[hist_start:],
        ts_raw.values[hist_start:],
        color="#999999",
        linewidth=1.0,
        alpha=0.5,
        label="Recent history",
    )

    def band90(res):
        half95 = (res["ci_upper"] - res["ci_lower"]) / 2
        center = res["forecast"]
        # convert 95% band (1.96σ) to 90% (1.645σ)
        factor = 1.645 / 1.96
        lower = center - half95 * factor
        upper = center + half95 * factor
        return lower, upper

    base_lower, base_upper = band90(base_result)
    calib_lower, calib_upper = band90(calibrated_result)

    # base in light gray, calibrated in vivid secondary
    ax.fill_between(
        forecast_dates,
        base_lower,
        base_upper,
        color="#d9d9d9",
        alpha=0.16,
        edgecolor="#888888",
        linewidth=0.8,
        label="Pre-calibration 90%",
    )
    ax.fill_between(
        forecast_dates,
        calib_lower,
        calib_upper,
        color=COLORS["secondary"],
        alpha=0.28,
        label="Post-calibration 90%",
    )
    ax.plot(
        forecast_dates,
        calibrated_result["forecast"],
        color=COLORS["secondary"],
        linewidth=2.8,
        linestyle="-",
        zorder=5,
        label="Post-calibration mean",
    )

    width_pre = np.median(base_upper - base_lower)
    width_post = np.median(calib_upper - calib_lower)
    if width_pre > 0:
        shrink = (1 - width_post / width_pre) * 100
        ax.text(
            forecast_dates[len(forecast_dates) // 2],
            np.min(calib_lower) * 0.98,
            f"Band shrink: {shrink:.0f}%",
            fontsize=10,
            color=COLORS["secondary"],
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["secondary"], alpha=0.8),
        )

    target_date = pd.to_datetime("2023-03-01")
    t_idx = (target_date - forecast_dates[0]).days
    if 0 <= t_idx < len(forecast_dates):
        point = calibrated_result["forecast"][t_idx]
        lo = calib_lower[t_idx]
        hi = calib_upper[t_idx]
        ax.scatter(target_date, point, color=COLORS["accent"], s=260, marker="*", edgecolors=COLORS["dark"], linewidths=2)
        ax.annotate(
            f"2023-03-01\nPoint: {point:,.0f}\n90%: [{lo:,.0f},{hi:,.0f}]",
            xy=(target_date, point),
            xytext=(30, 30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor=COLORS["accent"], linewidth=1.5),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark"], linewidth=1.5),
            fontsize=10,
            color=COLORS["dark"],
        )
    ax.set_title("Ensemble Fan Chart (Pre vs Post Calibration)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reported results")
    ax.legend(loc="upper right", framealpha=0.98, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_dir / "3_fan_calibration.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_reliability_sharpness(wf_initial: Dict, wf_calibrated: Dict, base_result: Dict, calibrated_result: Dict, output_dir: Path):
    """Reliability curve (pre/post calibration) + sharpness vs horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    horizons = sorted(set(list(wf_initial.keys()) + list(wf_calibrated.keys())))
    pre_cov = [wf_initial.get(h, {}).get("coverage", np.nan) for h in horizons]
    post_cov = [wf_calibrated.get(h, {}).get("coverage", np.nan) for h in horizons]
    axes[0].plot(horizons, pre_cov, marker="o", color=COLORS["primary"], label="Pre-calibration")
    axes[0].plot(horizons, post_cov, marker="o", color=COLORS["secondary"], label="Post-calibration")
    axes[0].axhline(90, color=COLORS["dark"], linestyle="--", linewidth=1.2, label="Nominal 90%")
    axes[0].set_title("Reliability: Nominal vs Actual Coverage", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Horizon (days)")
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_ylim(60, 100)
    axes[0].grid(True, alpha=0.3, linestyle=":")
    axes[0].legend()

    # convert stored 95% bands to 90% for fair comparison
    shrink = 1.645 / 1.96
    base_half = (base_result["ci_upper"] - base_result["ci_lower"]) / 2 * shrink
    cal_half = (calibrated_result["ci_upper"] - calibrated_result["ci_lower"]) / 2 * shrink
    h_range = np.arange(1, len(base_half) + 1)
    axes[1].plot(h_range, base_half, color=COLORS["primary"], alpha=0.7, label="Pre-calibration")
    axes[1].plot(h_range, cal_half, color=COLORS["secondary"], alpha=0.9, label="Post-calibration")
    axes[1].scatter([60], [cal_half[59]], color=COLORS["accent"], zorder=5, label="H=60 fallback")
    axes[1].set_title("Sharpness: Interval Half-width vs Horizon", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Horizon (days)")
    axes[1].set_ylabel("Interval half-width")
    axes[1].grid(True, alpha=0.3, linestyle=":")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "4_reliability_sharpness.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_model_comparison(ts_raw: pd.Series, ensemble_result: Dict, other_result: Dict, output_dir: Path,
                          history_days: int = 120, suffix: str = "", history_window: int | None = None):
    """
    Side-by-side forecast comparison between ensemble and another model (Prophet/Chronos).
    
    Args:
        other_result: Dict with keys "ds", "yhat", "yhat_lower", "yhat_upper"
    """
    forecast_dates = pd.date_range(
        start=ts_raw.index[-1] + pd.Timedelta(days=1),
        periods=len(ensemble_result["yhat"]),
        freq="D",
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    hist_start = max(0, len(ts_raw) - history_days)
    ax.plot(ts_raw.index[hist_start:], ts_raw.values[hist_start:], color="#777777", linewidth=1.6, alpha=0.6, label="History")

    # shrink ensemble 95% band to 90%
    ens_center = ensemble_result["yhat"]
    ens_half95 = (ensemble_result["yhat_upper"] - ensemble_result["yhat_lower"]) / 2
    shrink = 1.645 / 1.96
    ens_lower90 = ens_center - ens_half95 * shrink
    ens_upper90 = ens_center + ens_half95 * shrink

    ax.fill_between(
        forecast_dates,
        ens_lower90,
        ens_upper90,
        color=COLORS["secondary"],
        alpha=0.18,
        label="Ensemble 90% CI",
    )
    ax.plot(forecast_dates, ens_center, color=COLORS["secondary"], linewidth=2.4, label="Ensemble mean")

    # 处理其他模型（Prophet或Chronos）
    other_dates = pd.to_datetime(other_result["ds"])
    # 将95% CI转换为90% CI（如果需要）
    other_lower = other_result["yhat_lower"]
    other_upper = other_result["yhat_upper"]
    other_half95 = (other_upper - other_lower) / 2
    other_lower90 = other_result["yhat"] - other_half95 * shrink
    other_upper90 = other_result["yhat"] + other_half95 * shrink
    
    # 确定模型名称（从suffix推断）
    if "chronos" in suffix.lower():
        model_name = "Chronos"
        model_color = COLORS["accent"]
    else:
        model_name = "Prophet"
        model_color = COLORS["primary"]
    
    ax.fill_between(
        other_dates,
        other_lower90,
        other_upper90,
        color=model_color,
        alpha=0.14,
        label=f"{model_name} 90% CI",
    )
    ax.plot(other_dates, other_result["yhat"], color=model_color, linewidth=2.0, linestyle="--", label=f"{model_name} mean")

    target_date = pd.to_datetime("2023-03-01")
    if target_date in forecast_dates:
        idx = (target_date - forecast_dates[0]).days
        ens_pt = ensemble_result["yhat"][idx]
        ax.scatter(target_date, ens_pt, color=COLORS["accent"], marker="*", s=200, zorder=6, edgecolors=COLORS["dark"], linewidths=2)
        ax.annotate(
            "Target 2023-03-01",
            xy=(target_date, ens_pt),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["accent"], alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark"]),
            fontsize=9,
        )

    # 根据suffix和history_window设置不同的标题
    if "chronos" in suffix.lower():
        if suffix == "_chronos_main":
            if history_window:
                title = f"Ensemble vs Chronos Forecast (Recent {history_window} days)"
            else:
                title = "Ensemble vs Chronos Forecast (Main Window)"
        elif suffix == "_chronos_full":
            title = "Ensemble vs Chronos Forecast (Full History)"
        else:
            title = "Ensemble vs Chronos Forecast"
    else:
        if suffix == "_main":
            if history_window:
                title = f"Ensemble vs Prophet Forecast (Recent {history_window} days)"
            else:
                title = "Ensemble vs Prophet Forecast (Main Window)"
        elif suffix == "_full":
            title = "Ensemble vs Prophet Forecast (Full History)"
        else:
            title = "Ensemble vs Prophet Forecast (recent window)"
    
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Reported results")
    ax.legend(loc="upper right", framealpha=0.95, facecolor="white")
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    filename = f"5_model_comparison{suffix}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_three_way_comparison(ts_raw: pd.Series, ensemble_result: Dict, 
                              prophet_result: Dict | None, chronos_result: Dict | None,
                              output_dir: Path, history_days: int = 120, 
                              history_window: int | None = None, suffix: str = ""):
    """
    三模型对比可视化：Ensemble vs Prophet vs Chronos
    
    Args:
        ts_raw: 原始时间序列
        ensemble_result: Ensemble预测结果
        prophet_result: Prophet预测结果（可为None）
        chronos_result: Chronos预测结果（可为None）
        output_dir: 输出目录
        history_days: 历史数据展示天数
        history_window: 历史窗口长度
    """
    # 🔧 修复：ensemble_result 的格式是 {ds, yhat, yhat_lower, yhat_upper}
    forecast_dates = ensemble_result.get("ds", pd.date_range(
        start=ts_raw.index[-1] + pd.Timedelta(days=1),
        periods=len(ensemble_result.get("yhat", [])),
        freq="D",
    ))
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("white")

    # 历史数据
    hist_start = max(0, len(ts_raw) - history_days)
    ax.plot(ts_raw.index[hist_start:], ts_raw.values[hist_start:], 
            color="#777777", linewidth=1.8, alpha=0.7, label="Historical Data", zorder=1)

    # Ensemble预测（90% CI - 直接使用已计算好的区间）
    ens_center = ensemble_result["yhat"]  # 🔧 修复：使用 yhat
    ens_lower90 = ensemble_result["yhat_lower"]  # 🔧 修复：直接使用 yhat_lower (已经是90% CI)
    ens_upper90 = ensemble_result["yhat_upper"]  # 🔧 修复：直接使用 yhat_upper (已经是90% CI)

    ax.fill_between(
        forecast_dates,
        ens_lower90,
        ens_upper90,
        color=COLORS["secondary"],
        alpha=0.15,
        label="Ensemble 90% CI",
        zorder=2
    )
    ax.plot(forecast_dates, ens_center, color=COLORS["secondary"], 
            linewidth=2.6, label="Ensemble Point Forecast", zorder=3)

    # Prophet预测（如果可用，Prophet 默认返回 95% CI，需要缩小到 90%）
    if prophet_result is not None:
        prophet_dates = pd.to_datetime(prophet_result["ds"])
        prophet_lower = prophet_result["yhat_lower"]
        prophet_upper = prophet_result["yhat_upper"]
        prophet_half95 = (prophet_upper - prophet_lower) / 2
        shrink = 1.645 / 1.96  # 从 95% CI 缩小到 90% CI
        prophet_lower90 = prophet_result["yhat"] - prophet_half95 * shrink
        prophet_upper90 = prophet_result["yhat"] + prophet_half95 * shrink
        
        ax.fill_between(
            prophet_dates,
            prophet_lower90,
            prophet_upper90,
            color=COLORS["primary"],
            alpha=0.12,
            label="Prophet 90% CI",
            zorder=2
        )
        ax.plot(prophet_dates, prophet_result["yhat"], 
                color=COLORS["primary"], linewidth=2.2, 
                linestyle="--", label="Prophet Point Forecast", zorder=3)

    # Chronos预测（如果可用，Chronos 返回 90% CI，直接使用）
    if chronos_result is not None:
        chronos_dates = pd.to_datetime(chronos_result["ds"])
        chronos_lower90 = chronos_result["yhat_lower"]  # 🔧 直接使用 90% CI
        chronos_upper90 = chronos_result["yhat_upper"]  # 🔧 直接使用 90% CI
        
        ax.fill_between(
            chronos_dates,
            chronos_lower90,
            chronos_upper90,
            color=COLORS["accent"],
            alpha=0.12,
            label="Chronos 90% CI",
            zorder=2
        )
        ax.plot(chronos_dates, chronos_result["yhat"], 
                color=COLORS["accent"], linewidth=2.2, 
                linestyle=":", label="Chronos Point Forecast", zorder=3)

    # 目标日期标记
    target_date = pd.to_datetime("2023-03-01")
    if target_date in forecast_dates:
        idx = (target_date - forecast_dates[0]).days
        ens_pt = ensemble_result["yhat"][idx]
        ax.scatter(target_date, ens_pt, color=COLORS["warning"], marker="*", 
                  s=300, zorder=6, edgecolors=COLORS["dark"], linewidths=2.5)
        ax.annotate(
            "Target: 2023-03-01",
            xy=(target_date, ens_pt),
            xytext=(25, 25),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="white", 
                     edgecolor=COLORS["warning"], linewidth=2, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark"], linewidth=2),
            fontsize=11,
            fontweight="bold",
        )

    # 标题
    if history_window:
        title = f"Three-Model Comparison: Ensemble vs Prophet vs Chronos (History Window={history_window} days)"
    else:
        title = "Three-Model Comparison: Ensemble vs Prophet vs Chronos (Full History)"
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Reported Results", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.98, facecolor="white", 
             edgecolor="#cccccc", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle=":", zorder=0)
    
    plt.tight_layout()
    filename = f"6_three_way_comparison{suffix}.png"
    plt.savefig(output_dir / filename, dpi=150, 
               bbox_inches="tight", facecolor="white")
    plt.close()


def generate_unified_comparison_report(all_results: Dict, output_dir: Path):
    """
    生成统一的三模型对比报告（包含两种窗口）
    
    Args:
        all_results: 包含'full'和'240'两种窗口的对比结果
        output_dir: 输出目录
    """
    lines = [
        "=" * 70,
        "Unified Model Comparison Report: Ensemble vs Prophet vs Chronos",
        "=" * 70,
        "\nAll models use the same evaluation criteria:",
        "  - CV settings: min_train=180, n_splits=3, horizon=30",
        "  - Confidence interval: 95% CI (converted to 90% for visualization)",
        "  - Evaluation metrics: RMSE, Coverage, MASE, Pinball, Winkler",
        "\n" + "=" * 70,
    ]
    
    # 全量数据对比
    lines.append("\n【Full History Comparison】")
    comparison_results_full = all_results.get('full', {})
    
    if comparison_results_full.get('prophet', {}).get('metrics'):
        p = comparison_results_full['prophet']['metrics']
        lines.append("\nProphet:")
        lines.append(f"  RMSE:     {p['rmse']:.4f}")
        lines.append(f"  Coverage: {p['coverage']:.3f} (target: 0.95)")
        lines.append(f"  MASE:     {p['mase']:.4f} (lower is better)")
        lines.append(f"  Pinball:  {p['pinball']:.4f} (lower is better)")
        lines.append(f"  Winkler:  {p['winkler']:.2f} (lower is better)")
    
    if comparison_results_full.get('chronos', {}).get('metrics'):
        c = comparison_results_full['chronos']['metrics']
        lines.append("\nChronos-2 Transformer:")
        lines.append(f"  RMSE:     {c['rmse']:.4f}")
        lines.append(f"  Coverage: {c['coverage']:.3f} (target: 0.95)")
        lines.append(f"  MASE:     {c['mase']:.4f} (lower is better)")
        lines.append(f"  Pinball:  {c['pinball']:.4f} (lower is better)")
        lines.append(f"  Winkler:  {c['winkler']:.2f} (lower is better)")
    
    # 240天数据对比
    lines.append("\n" + "=" * 70)
    lines.append("【Recent 240 Days Comparison】")
    comparison_results_240 = all_results.get('240', {})
    
    if comparison_results_240.get('prophet', {}).get('metrics'):
        p = comparison_results_240['prophet']['metrics']
        lines.append("\nProphet:")
        lines.append(f"  RMSE:     {p['rmse']:.4f}")
        lines.append(f"  Coverage: {p['coverage']:.3f} (target: 0.95)")
        lines.append(f"  MASE:     {p['mase']:.4f} (lower is better)")
        lines.append(f"  Pinball:  {p['pinball']:.4f} (lower is better)")
        lines.append(f"  Winkler:  {p['winkler']:.2f} (lower is better)")
    
    if comparison_results_240.get('chronos', {}).get('metrics'):
        c = comparison_results_240['chronos']['metrics']
        lines.append("\nChronos-2 Transformer:")
        lines.append(f"  RMSE:     {c['rmse']:.4f}")
        lines.append(f"  Coverage: {c['coverage']:.3f} (target: 0.95)")
        lines.append(f"  MASE:     {c['mase']:.4f} (lower is better)")
        lines.append(f"  Pinball:  {c['pinball']:.4f} (lower is better)")
        lines.append(f"  Winkler:  {c['winkler']:.2f} (lower is better)")
    
    # 对比总结
    lines.append("\n" + "=" * 70)
    lines.append("Summary")
    lines.append("=" * 70)
    
    # 全量数据最佳模型
    if (comparison_results_full.get('prophet', {}).get('metrics') and 
        comparison_results_full.get('chronos', {}).get('metrics')):
        p = comparison_results_full['prophet']['metrics']
        c = comparison_results_full['chronos']['metrics']
        lines.append("\n[Full History] Best Model:")
        models = [("Prophet", p['rmse']), ("Chronos", c['rmse'])]
        best = min(models, key=lambda x: x[1])
        lines.append(f"  Best RMSE: {best[0]} ({best[1]:.4f})")
    
    # 240天数据最佳模型
    if (comparison_results_240.get('prophet', {}).get('metrics') and 
        comparison_results_240.get('chronos', {}).get('metrics')):
        p = comparison_results_240['prophet']['metrics']
        c = comparison_results_240['chronos']['metrics']
        lines.append("\n[Recent 240 Days] Best Model:")
        models = [("Prophet", p['rmse']), ("Chronos", c['rmse'])]
        best = min(models, key=lambda x: x[1])
        lines.append(f"  Best RMSE: {best[0]} ({best[1]:.4f})")
    
    lines.append("\nNote: Model architecture differences prevent perfectly fair comparison")
    lines.append("      - Ensemble: SARIMA-based with changepoint detection and log transform")
    lines.append("      - Prophet: Automatic changepoint/seasonality, uses log transform")
    lines.append("      - Chronos: Transformer architecture, no explicit feature engineering")
    lines.append("\nEnsemble metrics: See q1_final_clean.py output")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    (output_dir / "unified_comparison_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
