import sys
import os
import pandas as pd
import style_utils
import data_loader
import analysis_hard_mode
import analysis_difficulty
import analysis_heatmap
import predict_eerie  # <--- 导入新写的预测模块

# ==========================================
# 0. 工具类：双重日志记录器 (屏幕 + txt)
# ==========================================
class DualLogger:
    """
    将控制台输出同时保存到文件的工具类
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 设置日志保存路径
    output_file = "analysis_report.txt"
    sys.stdout = DualLogger(output_file)
    
    print(f"--- Analysis Started ---")
    print(f"Output will be saved to: {os.path.abspath(output_file)}\n")

    # 2. 应用莫兰迪画图风格
    style_utils.apply_morandi_style()

    # 3. 加载数据
    # 请确保这里使用的是包含新特征（词频、词性）的 CSV 文件
    data_path = 'data_final.csv' 
    print(f"Loading Data from: {data_path} ...")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found! Please run enrich_features.py first.")
        sys.exit(1)
        
    df = data_loader.load_and_clean_data(data_path)
    
    # 获取所有候选特征
    all_features = data_loader.get_feature_cols()
    
    # 4. 数据预处理：去除高度共线特征
    # 这一步保证了输入模型的数据是统计学上“干净”的
    features = data_loader.remove_collinear_features(df, all_features, threshold=0.90)
    print(f"Features ready for modeling: {len(features)} selected.")
    
    # 5. 按顺序执行所有分析任务
    try:
        # --- 任务一：Hard Mode 比例分析 (统计检验 + 归因) ---
        analysis_hard_mode.run_analysis_q1(df, features)
        
        # --- 任务二：难度预测与分级 (模型竞技场) ---
        analysis_difficulty.run_analysis_q2(df, features)

        # --- 可视化：绘制相关性热力图 (RQ1 & RQ2) ---
        analysis_heatmap.plot_correlation_heatmap(df, features)
        
        # --- 最终任务：预测单词 EERIE 的分布 ---
        # 直接调用新模块中的函数
        predict_eerie.run_prediction(df, features, target_word="EERIE")
        
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*50)
    print("✅ All Analyses Completed Successfully!")
    print(f"📝 Full Report saved to: {output_file}")
    print("="*50)