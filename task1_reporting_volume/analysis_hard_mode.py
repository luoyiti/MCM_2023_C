import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import style_utils
except ImportError:
    print("⚠️  警告: 未找到 style_utils，跳过绘图功能")
    style_utils = None

try:
    from data_loader import create_lag_features
except ImportError:
    # 如果找不到 data_loader，提供简单实现
    def create_lag_features(df, col, n_lags):
        df_lag = df.copy()
        lag_cols = []
        for i in range(1, n_lags + 1):
            lag_col = f"{col}_lag{i}"
            df_lag[lag_col] = df_lag[col].shift(i)
            lag_cols.append(lag_col)
        df_lag = df_lag.dropna()
        return df_lag, lag_cols

def run_analysis_q1(df, feature_cols):
    print("\n" + "="*40)
    print(">>> Question 1: Hard Mode Analysis")
    print("="*40)
    
    # 准备数据
    y = df['hard_mode_ratio']
    X = df[feature_cols]
    
    # 清理数据：移除 NaN 和 Inf
    valid_idx = y.dropna().index
    X, y = X.loc[valid_idx], y.loc[valid_idx]
    
    # 替换 inf 为 NaN，然后删除包含 NaN 的行
    X = X.replace([float('inf'), float('-inf')], float('nan'))
    valid_mask = ~(X.isna().any(axis=1))
    X, y = X[valid_mask], y[valid_mask]
    
    print(f"  ✓ 清理后数据: {len(X)} 行, {len(feature_cols)} 个特征")
    
    # --- Step 1: OLS 统计检验 ---
    print("\n--- Phase 1: OLS Statistical Test ---")
    X_const = sm.add_constant(X)
    ols = sm.OLS(y, X_const).fit()
    print(f"1. OLS R-squared: {ols.rsquared:.4f}")
    print("   (Interpretation: Low value < 0.1 means word attributes imply almost NO linear correlation)")
    
    # --- Step 2: Lasso 回归 (特征筛选验证) ---
    print("\n--- Phase 2: Lasso Feature Selection ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso(alpha=0.001, random_state=42)
    lasso.fit(X_scaled, y)
    non_zero_coefs = sum(lasso.coef_ != 0)
    print(f"2. Lasso Retained Features: {non_zero_coefs} / {len(feature_cols)}")
    print("   (Interpretation: If few features retained, attributes have weak predictive power)")
    
    # --- Step 3: 滞后特征分析 (Lag Analysis) ---
    print("\n--- Phase 3: Lag vs Attribute Importance ---")
    # 创建滞后特征 (Lag Features): 昨天的、前天的比例
    # 这步是为了证明惯性效应
    df_lag, lag_cols = create_lag_features(df, 'hard_mode_ratio', 3)
    
    # 合并特征：单词属性 + 滞后特征
    X_lag = df_lag[feature_cols + lag_cols]
    y_lag = df_lag['hard_mode_ratio']
    
    # 训练随机森林
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_lag, y_lag)
    
    # 获取并打印 Top 5 重要性
    importances = pd.Series(rf.feature_importances_, index=X_lag.columns).sort_values(ascending=False)
    print("\n[关键结果] Top 5 Feature Importances:")
    print(importances.head(5))
    
    # 绘图（如果 style_utils 可用）
    if style_utils:
        style_utils.plot_feature_importance(importances, "Feature Importance: Hard Mode Ratio (Lag vs Attributes)")


def analyze_hard_mode():
    """
    主函数：加载数据并运行 Hard Mode 分析
    这个函数被 run_task1.py 调用
    """
    print("\n" + "="*70)
    print("Hard Mode 分析")
    print("="*70)
    
    try:
        # 使用包含特征的 CSV 文件
        csv_path = Path(__file__).parent.parent / 'data' / 'mcm_processed_data.csv'
        
        # 加载数据
        print(f"📂 加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 计算 hard_mode_ratio（使用 CSV 的列名）
        df['hard_mode_ratio'] = df['number_in_hard_mode'] / df['number_of_reported_results']
        
        # 特征列：排除非特征列
        exclude_cols = [
            'date', 'contest_number', 'word', 
            'number_of_reported_results', 'number_in_hard_mode',
            '1_try', '2_tries', '3_tries', '4_tries', 
            '5_tries', '6_tries', '7_or_more_tries_x', 'sum',
            'hard_mode_ratio'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            print("⚠️  警告: 未找到单词属性特征列，分析可能不完整")
            print("   可用列:", list(df.columns))
            return
        
        print(f"✓ 找到 {len(feature_cols)} 个特征列")
        print(f"✓ 数据行数: {len(df)}")
        
        # 运行分析
        run_analysis_q1(df, feature_cols)
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()