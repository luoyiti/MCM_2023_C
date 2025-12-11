import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor 
import style_utils

def run_analysis_q2(df, feature_cols):
    print("\n" + "="*40)
    print(">>> Question 2: Difficulty Prediction")
    print("="*40)
    
    # 目标变量：平均猜测次数 (Average Guesses)
    y_avg = df['avg_guesses'] 
    X = df[feature_cols]
    
    # 切分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y_avg, test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # 模块一：模型竞技场 (Model Arena)
    # ---------------------------------------------------------
    print("\n--- Phase 1: Model Arena (Comparing Models) ---")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "SVR": SVR()
    }
    
    results = []
    # 用于保存训练好的随机森林模型，以便后续分析特征
    best_rf_model = None 
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 计算指标
        score_r2 = r2_score(y_test, y_pred)
        score_mse = mean_squared_error(y_test, y_pred)
        score_mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            'Model': name, 
            'R2': score_r2, 
            'MSE': score_mse, 
            'MAE': score_mae
        })
        
        # 保存随机森林模型
        if name == "Random Forest":
            best_rf_model = model
            print(f"\n[Random Forest Performance]")
            print(f"  R2 Score: {score_r2:.4f} (解释了 {score_r2:.1%} 的难度波动)")
            print(f"  MSE:      {score_mse:.4f}")
        
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    print("\nModel Arena Results:\n", results_df)
    style_utils.plot_model_comparison(results_df, metric='R2')
    
    # ---------------------------------------------------------
    # 模块二：【新增】特征贡献度分析 (Feature Importance)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Feature Importance Analysis (What makes a word hard?) ---")
    
    if best_rf_model:
        # 提取特征重要性
        importances = pd.Series(
            best_rf_model.feature_importances_, 
            index=feature_cols
        ).sort_values(ascending=False)
        
        print("\n[关键结果] 影响难度的 Top 10 特征:")
        print(importances.head(10))
        
        # 绘制特征重要性图
        style_utils.plot_feature_importance(
            importances, 
            "Top Factors Influencing Wordle Difficulty (RQ2)",
            top_n=10
        )
    
    # ---------------------------------------------------------
    # 模块三：K-Means 难度分级
    # ---------------------------------------------------------
    print("\n--- Phase 3: Difficulty Classification (K-Means) ---")
    dist_cols = ['1_try', '2_tries', '3_tries', '4_tries', '5_tries', '6_tries', '7_or_more_tries_x']
    X_dist = df[dist_cols].dropna()
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_dist)
    
    # 自动标记 Easy/Medium/Hard
    avg_guesses_cluster = np.dot(X_dist.values, np.array([1,2,3,4,5,6,7]))
    cluster_avg = [avg_guesses_cluster[labels == i].mean() for i in range(3)]
    sorted_idx = np.argsort(cluster_avg)
    label_map = {sorted_idx[0]: 'Easy', sorted_idx[1]: 'Medium', sorted_idx[2]: 'Hard'}
    
    print(f"Difficulty Clustering Map: {label_map}")