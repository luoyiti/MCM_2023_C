import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import style_utils 
from data_loader import create_lag_features # 确保导入这个函数

def run_analysis_q1(df, feature_cols):
    print("\n" + "="*40)
    print(">>> Question 1: Hard Mode Analysis")
    print("="*40)
    
    # 准备数据
    y = df['hard_mode_ratio']
    X = df[feature_cols]
    valid_idx = y.dropna().index
    X, y = X.loc[valid_idx], y.loc[valid_idx]
    
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
    
    # 绘图
    style_utils.plot_feature_importance(importances, "Feature Importance: Hard Mode Ratio (Lag vs Attributes)")