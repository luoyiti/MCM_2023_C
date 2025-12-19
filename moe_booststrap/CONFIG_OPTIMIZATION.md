# MoE 配置优化说明

## 优化目标
在保证预测准确度的同时，最大化2个专家之间的差异，使专家更清晰地分化。

## 关键配置调整（参照 Moe_Softmax.py 最佳实践）

### 1. 核心 MoE 参数

| 参数 | 原值 | 新值 | 理由 |
|------|------|------|------|
| `NUM_EXPERTS` | 3 | **2** | ✅ 小样本场景下2个专家更稳定，更容易形成明确的专家分化 |
| `TOP_K` | 2 | **1** | ✅ 形成"硬分群"效果，每个样本只路由到1个专家，最大化专家间差异 |
| `HIDDEN_SIZE` | 64 | **64** | ✅ 保持不变，经验证的最佳值 |
| `AUX_LOSS_WEIGHT` | 1e-3 | **1e-3** | ✅ 保持不变，已是最佳平衡点 |
| `EXPERT_DIVERSITY_COEF` | 1e-4 | **5e-4** | ✅ 提高5倍，更强地鼓励专家参数差异化 |

### 2. 🆕 专家输出差异化损失（核心改进）

**问题诊断：**
原有的 `expert_diversity_penalty` 只惩罚专家**参数**的相似度（余弦相似度），但参数不同并不能保证输出不同。

**解决方案：** 新增两种输出差异化损失函数：

| 损失函数 | 配置参数 | 默认值 | 作用 |
|----------|----------|--------|------|
| `expert_output_diversity_loss` | `EXPERT_OUTPUT_DIVERSITY_COEF` | **0.1** | 惩罚专家输出分布的余弦相似度 |
| `expert_js_divergence_loss` | `EXPERT_JS_DIVERGENCE_COEF` | **0.05** | 最大化专家输出分布的JS散度 |

**工作原理：**
```python
# 对每个batch：
# 1. 让所有专家独立计算输出（不经过门控）
# 2. 计算每对专家的平均输出分布
# 3. 计算pairwise相似度/散度作为损失
```

**总损失公式：**
```
loss = loss_main
     + aux_coef × aux_loss                     # 负载平衡
     + expert_diversity_coef × div_pen         # 参数差异化
     + expert_output_diversity_coef × output_div_loss   # 输出相似度惩罚
     + expert_js_divergence_coef × js_div_loss # 负JS散度（最小化=最大化差异）
```

### 3. 网格搜索范围调整

**原配置：**
```python
GRID_NUM_EXPERTS = [2, 3, 4, 6]
GRID_TOP_K = [1, 2, 3]
GRID_HIDDEN_SIZE = [32, 64, 128]
```

**新配置：**
```python
GRID_NUM_EXPERTS = [2, 3, 4]      # 聚焦2专家，减少6专家
GRID_TOP_K = [1, 2]                # 重点验证TOP_K=1
GRID_HIDDEN_SIZE = [64, 128]       # 聚焦最优区域，去掉32
```

**理由：** 减少搜索空间，聚焦于经验证有效的配置范围。

### 4. 专家分化搜索参数优化

**辅助损失系数网格：**
```python
# 原: [1e-3, 5e-3, 1e-2]
# 新: [5e-4, 1e-3, 2e-3]  # 针对2专家配置微调
```

**差异化系数网格：**
```python
# 原: [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
# 新: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]  # 探索更高的差异化强度
```

**理由：** 去掉0和过低值，探索更高的差异化系数范围。

## 预期效果

### ✅ 准确度保证
- **TOP_K=1 + NUM_EXPERTS=2** 在 Moe_Softmax.py 中已验证能保持良好准确度
- 硬分群虽然减少了灵活性，但在小样本场景下反而减少过拟合
- 2个专家足以捕捉主要的数据模式分组

### ✅ 专家差异最大化机制（4层机制）

1. **结构性差异（TOP_K=1）**
   - 每个样本只路由到1个专家，形成硬边界
   - 专家之间没有输出混合，各自独立负责不同样本集
   - 这是最强的专家分化机制

2. **参数差异化损失（EXPERT_DIVERSITY_COEF=5e-4）**
   - 惩罚专家参数向量的余弦相似度
   - 鼓励专家学习正交的特征表示
   - 提高5倍系数加强差异化压力

3. **🆕 输出相似度惩罚（EXPERT_OUTPUT_DIVERSITY_COEF=0.1）**
   - 直接计算专家输出分布的余弦相似度
   - 惩罚相似的输出分布，强制专家产生不同预测
   - 这是最直接的差异化手段

4. **🆕 JS散度最大化（EXPERT_JS_DIVERGENCE_COEF=0.05）**
   - 计算专家输出分布的JS散度
   - 通过最小化负JS散度来最大化差异
   - 从信息论角度鼓励分布差异

5. **负载平衡（AUX_LOSS_WEIGHT=1e-3）**
   - 防止所有样本塌陷到单一专家
   - 确保两个专家都得到充分训练
   - 维持约50:50的样本分配比例

## 可视化验证

使用 `plot_expert_parallel_coordinates()` 函数时，期望看到：

1. **上图（排名图）**：两条专家折线应该有明显交叉，表示在不同尝试次数上的排名互换
2. **排名距离矩阵**：两个专家的距离应该 > 3.0（最大值为√(7×4)≈5.29）
3. **排名稳定性**：如果某个专家在所有尝试次数上都保持同一排名，稳定性会很低（标准差大），说明专家有特定偏向

使用 `plot_sample_expert_decomposition()` 时，期望看到：
- 不同样本被路由到不同专家
- 同一个专家负责的样本应该有相似的分布特征
- 两个专家的预测分布形状应该有明显差异

## 调参建议

如果专家仍不够差异化，可以尝试：

```python
# 增加输出差异化强度
EXPERT_OUTPUT_DIVERSITY_COEF = 0.2  # 从0.1增加到0.2
EXPERT_JS_DIVERGENCE_COEF = 0.1     # 从0.05增加到0.1

# 注意：过高的差异化系数可能损害准确度
# 建议范围：
# EXPERT_OUTPUT_DIVERSITY_COEF: 0.05 ~ 0.3
# EXPERT_JS_DIVERGENCE_COEF: 0.02 ~ 0.15
```

## 使用方法

配置已自动更新，直接运行训练即可：

```python
from moe_booststrap import train_moe, evaluate
# ... 加载数据 ...
model, info = train_moe(X_train, P_train, X_val, P_val, ...)
```

或在 notebook 中重新训练：
```python
# 重新导入以应用新配置
import sys
for mod in list(sys.modules.keys()):
    if 'moe_booststrap' in mod:
        del sys.modules[mod]

from moe_booststrap import *
# ... 重新训练 ...
```

## 参考文献

配置优化参考自 `task2_distribution_prediction/models/Moe_Softmax.py`，该文件包含详细的：
- 超参数调优说明
- 模型架构解释  
- 损失函数设计原理
- 专家使用率分析方法

---
**更新时间：** 2025-12-18  
**优化目标：** 在保证准确度的同时最大化2个专家的差异
**核心改进：** 新增输出差异化损失，直接惩罚专家输出相似度
