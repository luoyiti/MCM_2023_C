# 专家分组分析模块使用指南

## 📌 概述

本模块已添加到 `moe_with_bootstrap.ipynb` notebook中，用于分析MoE模型的专家分配行为，并生成可视化结果。

## ⚠️ 重要提示：Top-K路由问题

**问题**：理论上有3个专家，但使用Top-K路由（TOP_K=2）时，只有2个专家会被分配样本。

**原因**：MoE模型的Top-K机制只选择权重最高的K个专家，导致排名第3的专家永远不会被使用。

**解决方案**：本模块提供了基于原始softmax概率的修正方案，确保所有3个专家都能被分配样本。详见下文"诊断与修正"部分。

## 🚀 快速开始

### 1. 运行前准备

确保已安装必要的依赖：

```bash
# 基础依赖（应该已安装）
pip install numpy pandas torch scikit-learn matplotlib scipy

# 词云生成依赖（可选，如不安装仍可运行其他分析）
pip install wordcloud
```

### 2. 运行顺序

在notebook中按以下顺序执行：

1. **训练模型** - 运行所有训练相关的单元格，直到得到最优模型

2. **原始专家分配** - 运行"专家分组分析与词云可视化"部分：
   - 第一个单元格：数据专家分配（会发现只有2个专家有数据）
   
3. **诊断与修正** - 运行"专家使用率诊断"部分：
   - 诊断单元格：分析为什么只有2个专家
   - 修正方案单元格：使用softmax概率重新分配
   - 词云生成单元格：生成修正后的词云
   - 预测分析单元格：分析修正后的预测分布

4. **查看结果** - 检查生成的文件和可视化

## 📊 生成的输出

所有输出文件保存在 `moe_bootstrap_output/` 目录：

### 原始方案的数据文件（Top-K路由）

| 文件 | 说明 |
|------|------|
| `expert_assignment_all_data.csv` | 每个词的专家分配（只有2个专家有数据） |

### 修正方案的数据文件（Softmax概率）

| 文件 | 说明 |
|------|------|
| `expert_assignment_fixed.csv` | 修正后的专家分配（3个专家都有数据）⭐ |
| `expert_feature_statistics_fixed.csv` | 修正后的各专家特征统计 |

### 可视化文件

| 文件 | 说明 |
|------|------|
| `expert_wordclouds.png` | 原始方案的词云图（可能只有2个） |
| `expert_wordclouds_fixed.png` | 修正后的词云图（3个专家）⭐ |
| `expert_feature_distributions.png` | 原始方案的特征分布 |
| `expert_prediction_distributions_fixed.png` | 修正后的预测分布图⭐ |

**推荐使用**：带 ⭐ 标记的修正版本文件

## 📋 CSV数据字段说明

### expert_assignment_all_data.csv

- **word**: 词汇
- **primary_expert**: 主要负责的专家ID (0, 1, 或 2)
- **expert_0_weight**: 专家0的门控权重
- **expert_1_weight**: 专家1的门控权重
- **expert_2_weight**: 专家2的门控权重
- **pred_1_try**: 预测1次尝试成功的概率
- **pred_2_tries**: 预测2次尝试成功的概率
- ... (其他尝试次数的预测概率)

## 🔍 分析要点

### 1. 专家分配

- 每个词根据门控网络的权重被分配给权重最大的专家
- `primary_expert` 列表示该词主要由哪个专家处理
- 三个专家的权重和为1（经过softmax归一化）

### 2. 词云图解读

- **词的大小**: 反映该词在专家门控中的权重
- **词的分布**: 展示每个专家"擅长"处理哪些类型的词
- 可以发现专家是否有明显的专业化方向

### 3. 特征分析

选择6个关键特征进行对比：
- Zipf-value: 词频
- letter_entropy: 字母熵
- feedback_entropy: 反馈熵
- scrabble_score: 拼字得分
- hamming_neighbors: 汉明距离邻居数
- semantic_distance: 语义距离

通过对比可以发现：
- 专家0可能专注于高频词
- 专家1可能处理中等难度词
- 专家2可能负责低频/难词

### 4. 预测分布

- 展示每个专家的平均预测倾向
- 可以看出哪个专家倾向于预测更高/更低的成功率
- 帮助理解专家的"性格"差异

## 💡 使用建议

1. **首次运行**: 先运行完整个notebook，确保模型训练完成
2. **快速查看**: 只关注词云图可以快速了解专家分化情况
3. **深度分析**: 结合特征统计CSV进行定量分析
4. **模型调优**: 如果专家分化不明显，可以调整超参数：
   - 增大 `EXPERT_DIVERSITY_COEF` 鼓励专家差异化
   - 调整 `NUM_EXPERTS` 改变专家数量
   - 修改 `TOP_K` 影响专家路由策略

## ⚠️ 注意事项

1. **内存使用**: 处理全部数据时可能需要较大内存
2. **词云库**: 如未安装wordcloud，会跳过词云生成但继续其他分析
3. **字体问题**: 词云默认使用系统字体，如显示异常可能需要指定中文字体路径
4. **数据路径**: 确保 `DATA_PATH` 指向正确的数据文件

## 🐛 常见问题

**Q: 为什么只有2个专家有数据？**
A: 这是MoE模型的Top-K路由机制导致的。当`TOP_K=2`且`NUM_EXPERTS=3`时，排名第3的专家永远不会被选中。请使用notebook中的"修正方案"单元格来解决这个问题。

**Q: 修正方案和原始方案有什么区别？**
A: 
- 原始方案：使用Top-K过滤后的gates（实际推理时的行为）
- 修正方案：使用softmax概率（门控网络的原始输出，确保所有专家都被考虑）
- 推荐使用修正方案进行分析，以全面了解所有专家的特性

**Q: 如何从根本上解决专家分配不均的问题？**
A: 
1. 增大`EXPERT_DIVERSITY_COEF`（如改为1e-2）鼓励专家差异化
2. 设置`TOP_K=3`允许所有专家参与
3. 重新训练模型
注意：需要在训练阶段就做这些调整

**Q: 词云图无法生成？**
A: 安装wordcloud库: `pip install wordcloud`

**Q: 专家分配不均衡？**
A: 这可能是正常现象，说明某些专家更"通用"。使用修正方案可以看到基于门控概率的真实分配。

**Q: CSV文件太大？**
A: 如果只需要部分数据，可以在保存前筛选需要的列。

## 📧 反馈

如有问题或建议，请查看notebook中的详细注释或联系开发者。

---

## 🔬 技术细节：Top-K路由机制

### 问题根源

MoE模型的Top-K路由工作流程：

```python
# 1. 计算门控logits
logits = X @ w_gate

# 2. Softmax得到概率
probs = softmax(logits)  # shape: [batch_size, num_experts]

# 3. 选择Top-K（问题所在！）
top_k_probs, top_k_indices = probs.topk(k=TOP_K)

# 4. 重新归一化Top-K
top_k_gates = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)

# 5. 创建稀疏gates（其他专家权重为0）
gates = scatter(top_k_gates, top_k_indices)
```

**结果**：当`TOP_K < NUM_EXPERTS`时，某些专家的权重会被强制置为0！

### 两种分配策略对比

| 特性 | Top-K路由（原始） | Softmax概率（修正） |
|------|------------------|-------------------|
| 实现 | `gates.argmax()` | `full_probs.argmax()` |
| 特点 | 只考虑Top-K专家 | 考虑所有专家 |
| 专家覆盖 | 可能不完整 | 完整覆盖 |
| 与推理一致性 | ✓ 高 | 仅用于分析 |
| 推荐用途 | 理解实际路由 | 全面分析专家 |

### 长期解决方案

如果希望所有专家都被实际使用，需要在**训练时**调整：

```python
# 选项1：增大专家分化正则项
EXPERT_DIVERSITY_COEF = 1e-2  # 鼓励专家差异化

# 选项2：允许所有专家参与
TOP_K = NUM_EXPERTS  # 但会增加计算开销

# 选项3：增大负载平衡损失
AUX_COEF = 1e-2  # 鼓励专家负载均衡
```

然后重新训练模型即可。
