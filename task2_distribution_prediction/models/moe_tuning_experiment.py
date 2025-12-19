#!/usr/bin/env python3
"""
MoE模型参数调优实验脚本

目标：
1. 提升预测性能（降低MAE、KL散度，提升R²等）
2. 增强专家区分度（提升JS距离）
3. 在不降低性能的前提下增加专家数量和TOP_K

实验策略：
- 阶段1：基线测试（当前配置）
- 阶段2：专家数量扩展实验（2→4→6→8专家）
- 阶段3：TOP_K扩展实验（1→2→3）
- 阶段4：综合优化（最佳专家数+TOP_K+超参数调优）
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# 确保能导入原始模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 导入原始模块的所有函数和类
from moe_with_bootstrap import *

@dataclass
class ExperimentConfig:
    """实验配置"""
    num_experts: int
    hidden_size: int
    top_k: int
    aux_coef: float
    expert_diversity_coef: float
    lr: float
    weight_decay: float
    max_epochs: int
    patience: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_experts': self.num_experts,
            'hidden_size': self.hidden_size,
            'top_k': self.top_k,
            'aux_coef': self.aux_coef,
            'expert_diversity_coef': self.expert_diversity_coef,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'patience': self.patience
        }

@dataclass
class ExperimentResult:
    """实验结果"""
    config: ExperimentConfig
    metrics: Dict[str, float]
    expert_js_separation: float
    training_time: float
    best_epoch: int
    expert_usage_balance: float  # 专家使用均衡度（方差越小越均衡）
    
    def get_composite_score(self, alpha=0.7, beta=0.3) -> float:
        """综合评分：alpha*性能分数 + beta*专家分化分数"""
        # 性能分数（越小越好的指标取负数）
        perf_score = (
            -self.metrics.get('mae', 1.0) * 10 +  # MAE权重最高
            -self.metrics.get('kl', 1.0) * 5 +
            -self.metrics.get('js_mean', 1.0) * 3 +
            self.metrics.get('cos_sim', 0.0) * 2 +
            self.metrics.get('r2', 0.0) * 3
        )
        
        # 专家分化分数（JS距离越大越好，使用均衡度惩罚）
        expert_score = self.expert_js_separation * (1 - self.expert_usage_balance)
        
        return alpha * perf_score + beta * expert_score

class MoETuningExperiment:
    """MoE调优实验管理器"""
    
    def __init__(self, output_dir: str = "moe_tuning_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据（只加载一次）
        print("加载数据...")
        self.data = self._load_data()
        
        # 实验结果存储
        self.results: List[ExperimentResult] = []
        
    def _load_data(self) -> Tuple:
        """加载并预处理数据"""
        return load_and_split_data()
    
    def _train_and_evaluate_config(self, config: ExperimentConfig) -> ExperimentResult:
        """训练并评估单个配置"""
        print(f"\n{'='*60}")
        print(f"实验配置: {config.num_experts}专家, TOP_K={config.top_k}, hidden={config.hidden_size}")
        print(f"{'='*60}")
        
        # 解包数据
        X_train, X_val, X_test, P_train, P_val, P_test, N_train, N_val, N_test, holdout_pack = self.data
        
        # 计算样本权重
        Wtr = (
            torch.tensor(make_weights_from_N(N_train, WEIGHT_MODE), device=DEVICE)
            if N_train is not None else None
        )
        Wva = (
            torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE)
            if N_val is not None else None
        )
        
        # 临时修改全局参数
        global NUM_EXPERTS, HIDDEN_SIZE, TOP_K, AUX_COEF, EXPERT_DIVERSITY_COEF
        global LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE
        
        old_params = {
            'NUM_EXPERTS': NUM_EXPERTS,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'TOP_K': TOP_K,
            'AUX_COEF': AUX_COEF,
            'EXPERT_DIVERSITY_COEF': EXPERT_DIVERSITY_COEF,
            'LR': LR,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'MAX_EPOCHS': MAX_EPOCHS,
            'PATIENCE': PATIENCE
        }
        
        # 设置新参数
        NUM_EXPERTS = config.num_experts
        HIDDEN_SIZE = config.hidden_size
        TOP_K = config.top_k
        AUX_COEF = config.aux_coef
        EXPERT_DIVERSITY_COEF = config.expert_diversity_coef
        LR = config.lr
        WEIGHT_DECAY = config.weight_decay
        MAX_EPOCHS = config.max_epochs
        PATIENCE = config.patience
        
        try:
            # 训练模型
            start_time = time.time()
            model, info = train_moe_with_params(
                X_train=X_train,
                P_train=P_train,
                X_val=X_val,
                P_val=P_val,
                Wtr=Wtr,
                Wva=Wva,
                num_experts=config.num_experts,
                hidden_size=config.hidden_size,
                top_k=config.top_k,
                aux_coef=config.aux_coef,
                expert_diversity_coef=config.expert_diversity_coef
            )
            training_time = time.time() - start_time
            
            # 评估模型
            P_pred, metrics = evaluate(model, X_test, P_test)
            
            # 计算专家分化指标
            expert_js_sep = expert_output_separation_js(model, X_test)
            
            # 计算专家使用均衡度
            model.eval()
            Xte = torch.tensor(X_test, device=DEVICE)
            with torch.no_grad():
                gates, _ = model.noisy_top_k_gating(Xte, train=False)
                gates_np = gates.cpu().numpy()
            
            expert_usage = gates_np.mean(axis=0)
            expert_usage_balance = float(np.var(expert_usage))  # 方差越小越均衡
            
            result = ExperimentResult(
                config=config,
                metrics=metrics,
                expert_js_separation=expert_js_sep,
                training_time=training_time,
                best_epoch=info['best_epoch'],
                expert_usage_balance=expert_usage_balance
            )
            
            print(f"训练完成 - 时间: {training_time:.1f}s, 最佳轮次: {info['best_epoch']}")
            print(f"性能指标 - MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            print(f"专家分化 - JS距离: {expert_js_sep:.4f}, 使用均衡度: {expert_usage_balance:.4f}")
            print(f"综合评分: {result.get_composite_score():.4f}")
            
            return result
            
        finally:
            # 恢复原始参数
            NUM_EXPERTS = old_params['NUM_EXPERTS']
            HIDDEN_SIZE = old_params['HIDDEN_SIZE']
            TOP_K = old_params['TOP_K']
            AUX_COEF = old_params['AUX_COEF']
            EXPERT_DIVERSITY_COEF = old_params['EXPERT_DIVERSITY_COEF']
            LR = old_params['LR']
            WEIGHT_DECAY = old_params['WEIGHT_DECAY']
            MAX_EPOCHS = old_params['MAX_EPOCHS']
            PATIENCE = old_params['PATIENCE']
    
    def run_baseline_experiment(self) -> ExperimentResult:
        """运行基线实验"""
        print("\n" + "="*80)
        print("阶段1: 基线实验")
        print("="*80)
        
        baseline_config = ExperimentConfig(
            num_experts=2,
            hidden_size=64,
            top_k=1,
            aux_coef=1e-3,
            expert_diversity_coef=1e-4,
            lr=5e-3,
            weight_decay=1e-4,
            max_epochs=500,
            patience=50
        )
        
        result = self._train_and_evaluate_config(baseline_config)
        self.results.append(result)
        return result
    
    def run_expert_scaling_experiments(self, expert_counts: List[int] = [3, 4, 6, 8]) -> List[ExperimentResult]:
        """运行专家数量扩展实验"""
        print("\n" + "="*80)
        print("阶段2: 专家数量扩展实验")
        print("="*80)
        
        results = []
        base_config = ExperimentConfig(
            num_experts=2,  # 会被覆盖
            hidden_size=64,
            top_k=1,
            aux_coef=1e-3,
            expert_diversity_coef=1e-4,
            lr=5e-3,
            weight_decay=1e-4,
            max_epochs=400,  # 稍微减少轮次加速实验
            patience=40
        )
        
        for num_experts in expert_counts:
            config = ExperimentConfig(
                num_experts=num_experts,
                hidden_size=base_config.hidden_size,
                top_k=base_config.top_k,
                aux_coef=base_config.aux_coef,
                expert_diversity_coef=base_config.expert_diversity_coef,
                lr=base_config.lr,
                weight_decay=base_config.weight_decay,
                max_epochs=base_config.max_epochs,
                patience=base_config.patience
            )
            
            result = self._train_and_evaluate_config(config)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_topk_scaling_experiments(self, topk_values: List[int] = [2, 3]) -> List[ExperimentResult]:
        """运行TOP_K扩展实验"""
        print("\n" + "="*80)
        print("阶段3: TOP_K扩展实验")
        print("="*80)
        
        results = []
        # 使用目前最佳的专家数量
        best_expert_count = self._get_best_expert_count()
        
        base_config = ExperimentConfig(
            num_experts=best_expert_count,
            hidden_size=64,
            top_k=1,  # 会被覆盖
            aux_coef=1e-3,
            expert_diversity_coef=1e-4,
            lr=5e-3,
            weight_decay=1e-4,
            max_epochs=400,
            patience=40
        )
        
        for top_k in topk_values:
            # TOP_K不能超过专家数量
            if top_k >= best_expert_count:
                continue
                
            config = ExperimentConfig(
                num_experts=base_config.num_experts,
                hidden_size=base_config.hidden_size,
                top_k=top_k,
                aux_coef=base_config.aux_coef,
                expert_diversity_coef=base_config.expert_diversity_coef,
                lr=base_config.lr,
                weight_decay=base_config.weight_decay,
                max_epochs=base_config.max_epochs,
                patience=base_config.patience
            )
            
            result = self._train_and_evaluate_config(config)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_hyperparameter_optimization(self) -> List[ExperimentResult]:
        """运行超参数优化实验"""
        print("\n" + "="*80)
        print("阶段4: 超参数优化实验")
        print("="*80)
        
        results = []
        best_config = self._get_best_config()
        
        # 超参数搜索网格
        param_grids = {
            'hidden_size': [64, 96, 128],
            'aux_coef': [5e-4, 1e-3, 2e-3, 5e-3],
            'expert_diversity_coef': [1e-5, 1e-4, 5e-4, 1e-3],
            'lr': [3e-3, 5e-3, 7e-3],
            'weight_decay': [5e-5, 1e-4, 2e-4]
        }
        
        # 贪心搜索：每次只调整一个参数
        current_config = best_config
        
        for param_name, param_values in param_grids.items():
            print(f"\n优化参数: {param_name}")
            best_score = current_config and self._get_result_by_config(current_config).get_composite_score() or -float('inf')
            best_param_config = current_config
            
            for param_value in param_values:
                # 创建新配置
                config_dict = current_config.to_dict() if current_config else best_config.to_dict()
                config_dict[param_name] = param_value
                
                test_config = ExperimentConfig(**config_dict)
                test_config.max_epochs = 300  # 减少轮次加速搜索
                test_config.patience = 30
                
                result = self._train_and_evaluate_config(test_config)
                results.append(result)
                self.results.append(result)
                
                score = result.get_composite_score()
                if score > best_score:
                    best_score = score
                    best_param_config = test_config
                    print(f"  新最佳 {param_name}={param_value}, 评分: {score:.4f}")
            
            current_config = best_param_config
        
        return results
    
    def _get_best_expert_count(self) -> int:
        """获取目前最佳的专家数量"""
        if not self.results:
            return 2
        
        expert_results = [r for r in self.results if r.config.top_k == 1]
        if not expert_results:
            return 2
        
        best_result = max(expert_results, key=lambda x: x.get_composite_score())
        return best_result.config.num_experts
    
    def _get_best_config(self) -> ExperimentConfig:
        """获取目前最佳的配置"""
        if not self.results:
            return ExperimentConfig(2, 64, 1, 1e-3, 1e-4, 5e-3, 1e-4, 500, 50)
        
        best_result = max(self.results, key=lambda x: x.get_composite_score())
        return best_result.config
    
    def _get_result_by_config(self, config: ExperimentConfig) -> ExperimentResult:
        """根据配置获取结果"""
        for result in self.results:
            if (result.config.num_experts == config.num_experts and
                result.config.top_k == config.top_k and
                result.config.hidden_size == config.hidden_size):
                return result
        return None
    
    def save_results(self):
        """保存实验结果"""
        # 保存详细结果
        results_data = []
        for result in self.results:
            data = {
                'config': result.config.to_dict(),
                'metrics': result.metrics,
                'expert_js_separation': result.expert_js_separation,
                'training_time': result.training_time,
                'best_epoch': result.best_epoch,
                'expert_usage_balance': result.expert_usage_balance,
                'composite_score': result.get_composite_score()
            }
            results_data.append(data)
        
        with open(os.path.join(self.output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 保存汇总表格
        summary_data = []
        for result in self.results:
            row = {
                'num_experts': result.config.num_experts,
                'hidden_size': result.config.hidden_size,
                'top_k': result.config.top_k,
                'aux_coef': result.config.aux_coef,
                'expert_diversity_coef': result.config.expert_diversity_coef,
                'lr': result.config.lr,
                'weight_decay': result.config.weight_decay,
                'mae': result.metrics['mae'],
                'kl': result.metrics['kl'],
                'js_mean': result.metrics['js_mean'],
                'cos_sim': result.metrics['cos_sim'],
                'r2': result.metrics['r2'],
                'expert_js_separation': result.expert_js_separation,
                'expert_usage_balance': result.expert_usage_balance,
                'training_time': result.training_time,
                'best_epoch': result.best_epoch,
                'composite_score': result.get_composite_score()
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('composite_score', ascending=False)
        df.to_csv(os.path.join(self.output_dir, 'experiment_summary.csv'), index=False)
        
        print(f"\n实验结果已保存到: {self.output_dir}/")
        print(f"最佳配置:")
        best_result = max(self.results, key=lambda x: x.get_composite_score())
        print(f"  专家数: {best_result.config.num_experts}")
        print(f"  TOP_K: {best_result.config.top_k}")
        print(f"  隐藏层: {best_result.config.hidden_size}")
        print(f"  综合评分: {best_result.get_composite_score():.4f}")
        print(f"  MAE: {best_result.metrics['mae']:.4f}")
        print(f"  R²: {best_result.metrics['r2']:.4f}")
        print(f"  专家JS距离: {best_result.expert_js_separation:.4f}")
    
    def run_full_experiment(self):
        """运行完整的调优实验"""
        print("开始MoE模型调优实验...")
        print(f"输出目录: {self.output_dir}")
        
        # 阶段1: 基线实验
        self.run_baseline_experiment()
        
        # 阶段2: 专家数量扩展
        self.run_expert_scaling_experiments([3, 4, 6, 8])
        
        # 阶段3: TOP_K扩展
        self.run_topk_scaling_experiments([2, 3])
        
        # 阶段4: 超参数优化
        self.run_hyperparameter_optimization()
        
        # 保存结果
        self.save_results()
        
        print("\n" + "="*80)
        print("实验完成！")
        print("="*80)

def main():
    """主函数"""
    set_seed(RANDOM_SEED)
    
    experiment = MoETuningExperiment()
    experiment.run_full_experiment()

if __name__ == "__main__":
    main()