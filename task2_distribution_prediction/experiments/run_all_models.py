"""
主程序入口 - 运行所有模型的训练和评估
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forcasting import config
from forcasting.forecasting_models import (
    load_data,
    train_and_evaluate_model,
    compare_models,
)
from forcasting.distribution_models import (
    load_distribution_data,
    train_and_evaluate_distribution_model,
    compare_distribution_models,
    generate_distribution_report,
)

def main():
    """主函数 - 运行完整的模型训练和评估流程"""
    
    print("=" * 80)
    print(" " * 20 + "Wordle 难度预测模型训练")
    print("=" * 80)
    
    # ========================================================================
    # 第一部分：回归模型（预测 autoencoder_value）
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("第一部分：回归模型训练")
    print("=" * 80)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()
    
    # 训练所有回归模型
    models_to_train = ['lasso', 'ridge', 'elasticnet', 'mlp', 'randomforest']
    all_results = []
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"训练 {model_name.upper()} 模型...")
        print(f"{'='*60}")
        
        result = train_and_evaluate_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv_splits=5,
            show_plots=False,  # 批量运行时不显示图表
            save_plots=True
        )
        all_results.append(result)
    
    # 模型对比
    print(f"\n{'='*60}")
    print("回归模型对比")
    print(f"{'='*60}")
    compare_models(all_results, save_path=f"{config.RESULTS_BASE}/models_comparison.png")
    
    # ========================================================================
    # 第二部分：分布预测模型（预测 7 维概率分布）
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("第二部分：分布预测模型训练")
    print("=" * 80)
    
    # 加载分布预测数据
    data_dict = load_distribution_data()
    
    # 训练 Linear-Softmax
    print(f"\n{'='*60}")
    print("训练 Linear-Softmax 模型...")
    print(f"{'='*60}")
    
    linear_results = train_and_evaluate_distribution_model(
        model_type='linear',
        data=data_dict,
        epochs=500,
        lr=0.01,
        patience=30,
        output_dir=config.LINEAR_SOFTMAX_RESULTS,
        save_plots=True,
        show_plots=False
    )
    
    # 训练 MLP-Softmax
    print(f"\n{'='*60}")
    print("训练 MLP-Softmax 模型...")
    print(f"{'='*60}")
    
    mlp_results = train_and_evaluate_distribution_model(
        model_type='mlp',
        data=data_dict,
        epochs=500,
        lr=0.01,
        patience=30,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        output_dir=config.MLP_SOFTMAX_RESULTS,
        save_plots=True,
        show_plots=False
    )
    
    # 分布模型对比
    print(f"\n{'='*60}")
    print("分布预测模型对比")
    print(f"{'='*60}")
    
    dist_results = [linear_results, mlp_results]
    compare_distribution_models(
        dist_results,
        output_dir=config.DISTRIBUTION_RESULTS,
        save=True,
        show=False
    )
    
    # 生成综合报告
    generate_distribution_report(
        dist_results,
        data=data_dict,
        output_path=f"{config.DISTRIBUTION_RESULTS}/comparison_report.txt"
    )
    
    # ========================================================================
    # 总结
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    
    print("\n回归模型结果：")
    for result in all_results:
        print(f"  {result['model_name']:15s} - Test R²: {result['test_metrics']['r2']:.4f}, "
              f"RMSE: {result['test_metrics']['rmse']:.4f}")
    
    print("\n分布预测模型结果：")
    for result in dist_results:
        print(f"  {result['model_name']:15s} - Test MAE: {result['test_metrics']['mae']:.4f}, "
              f"Cosine Sim: {result['test_metrics']['cosine_similarity']:.4f}")
    
    print(f"\n所有结果已保存到各自的输出目录")
    print("=" * 80)


if __name__ == "__main__":
    # 确保所有输出目录存在
    for dir_path in [
        config.LASSO_RESULTS,
        config.RIDGE_RESULTS,
        config.ELASTICNET_RESULTS,
        config.MLP_RESULTS,
        config.RANDOMFOREST_RESULTS,
        config.LINEAR_SOFTMAX_RESULTS,
        config.MLP_SOFTMAX_RESULTS,
        config.DISTRIBUTION_RESULTS,
    ]:
        config.ensure_dir(dir_path)
    
    # 运行主程序
    main()
