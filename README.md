# TNCN: Temporal Neighbor Co-occurrence Network

**第六届计图人工智能挑战赛 赛道一**（时序二分图 · 目标链接预测）

- **战队**:算法小分队（郑州大学）
- **最终成绩**:复赛 **25 名**（赛道一）
- **注**:随源码附带的原技术报告（PDF）标注排名为 51 名；复赛最终排名为 25 名（赛道一）。

---

## 1. 比赛与赛道简介

本赛道要求在**时序二分图（bipartite graph）**上进行目标链接预测（link prediction）。给定用户（`src`）与商品（`dst`）之间的历史交互序列（含时间戳），预测在某一时刻用户与候选商品之间发生交互的概率。每个测试样本给定 100 个候选商品，需输出每个候选的得分。官方评价指标为 **MRR（Mean Reciprocal Rank）**。

比赛数据分为 `dataset1` 与 `dataset2` 两个数据集，代码与超参按数据集各自调优（见第 4 节）。

## 2. 仓库结构

```
.
├── code/
│   ├── main_tncn.py     # 主训练 + 测试脚本（训练 3 个 epoch，逐 epoch 保存预测）
│   ├── model_tncn.py    # TNCNModel 模型定义（时序编码器 / TokenMixer / 交叉注意力 / CN）
│   ├── ii_graph.py      # II（item-item）共现图构建
│   └── ensemble.py      # 跨 epoch 排名集成（rank-based ensemble）
├── requirements.txt     # 依赖清单（含系统级说明）
├── LICENSE              # MIT
└── README.md            # 本文档（原技术报告全文并入）
```

## 3. TNCN 模型架构

TNCN 是基于官方基线 CRAFT（Causal Anonymous Walks）改进的模型。核心思想：利用时序邻居编码器捕捉用户的历史交互模式，通过交叉注意力机制让候选商品参考用户的历史行为序列，并结合 II/UI 图传播、共同邻居（CN）等结构信息进行综合判断。

### 3.1 时间编码

采用 `TimeBucketEmbed` 将时间差划分到自定义的 log 等差时间桶，然后嵌入到 `hidden_size` 维度。另有 `TimeProjection` 通过 MLP 将单维度特征投影到高维空间。

### 3.2 时序邻居编码器

使用 `jittor_geometric` 的 `get_historical_neighbors_left` 获取 `src` 的最近历史邻居（`num_neighbors - 1` 个），并在序列头部 prepend `src` 自身作为第 0 位置（表示当前用户身份）。

### 3.3 II/UI 图传播

- **II（item-item）图**：基于交互序列中相邻商品的共现关系构建有向图，采用注意力权重或静态权重传播（`ii_graph.py`）。
- **UI（user-item）图**：采用 LightGCN 风格的对称归一化图传播。

两者并行计算（FREEDOM 风格），最终表示为 `ui_out + ii_out - raw_emb`。

### 3.4 序列编码器（TokenMixer）

采用 MLP-Mixer 风格的 `TokenMixerLayer` 对 `src` 邻居序列进行编码，包含 channel mixing 与 token interaction 两个步骤，复杂度为 O(LD) 而非 O(L²D)。

### 3.5 交叉注意力

候选 `dst` 作为 query、`src` 的历史邻居序列作为 key/value，通过多头交叉注意力获取候选与用户历史行为的关联。注意力架构继承自 CRAFT。

### 3.6 共同邻居特征（CN）

计算 `src` 邻居与每个候选 `dst` 邻居的交集，通过嵌入的加权平均获得 CN 向量，类似 NCN 结构。

### 3.7 特征混合与输出

将 `attn_out`、`cn_features`、`src_emb_expand`、`src_xij` 等多个特征堆叠为特征序列，经 `FeatTokenMixer` 混合后通过 `output_layer`（MLP）输出得分。

## 4. 超参数配置

### 4.1 dataset1

- `hidden_size=128, n_layers=2, n_heads=2, dropout=0.2`
- `ii_layers=5, ii_use_attn=True, ii_attn_tau=2`
- `ui_layers=2, use_seq_self_attn=True, seq_encoder_layers=1`
- `use_feat_mixer=True, use_pop_feat=False`
- `use_infonce=True, lambda_infonce=0.1, infonce_temperature=0.07`
- `output_cat_repeat_times=True`
- `num_neighbors=128, test_num_neighbors=256, val_ratio=0.01, batch_size=256`
- `lr=0.0002, cosine scheduler, eta_min=0.00002`

### 4.2 dataset2

- `hidden_size=64, n_layers=2, n_heads=2, dropout=0.2`
- `ii_layers=3, ii_use_attn=True, ii_attn_tau=1`
- `ui_layers=2, use_seq_self_attn=True, seq_encoder_layers=1`
- `use_feat_mixer=True, use_pop_feat=False`
- `use_infonce=False, lambda_infonce=0.02`
- `output_cat_repeat_times=False`
- `num_neighbors=128, test_num_neighbors=256, val_ratio=0.01, batch_size=256`
- `lr=0.0002, cosine scheduler, eta_min=0.00002`

## 5. 训练细节

- 优化器：Adam，学习率 2e-4
- 学习率调度：Cosine Warmup Scheduler，热身 1 个 epoch
- 负样本采样：每正样本配 99 个负样本（`neg_sampling_ratio=99.0`）
- 损失函数：CrossEntropyLoss（100 分类）+ 可选的 InfoNCE + L2 正则化
- Early Stopping：patience=2
- 验证集切分：按时间序前 99% 训练，后 1% 验证
- 随机种子：2026，确保可复现

## 6. 排名集成方法

`dataset1` 与 `dataset2` 均训练 **3 个 epoch**。对每个数据集，将**第 1 个 epoch** 与**第 3 个 epoch** 的测试预测结果进行**排名集成**，得到该数据集的最终预测结果。

排名集成的具体步骤（对应 `code/ensemble.py` 的 `rank` 方法）：

1. 加载两个 epoch 的预测 CSV（无表头，每行 100 个候选得分）；
2. 对每个 epoch 的每行得分在列内排序得到排名（得分越高排名越靠前）；
3. 对两个 epoch 的排名取平均（`avg_rank`）；
4. 将平均排名转换为得分：`score = 1 / (avg_rank + 1)`；
5. 输出最终预测结果 CSV（无表头）。

> 优势：排名集成对得分的绝对大小不敏感，可以有效结合不同时刻模型的排序能力，而不受各模型得分分布差异的影响。

`ensemble.py` 使用 fire 支持命令行指定数据集与两个 epoch 的分数（**score1 对应第 1 个 epoch，score2 对应第 3 个 epoch**），并自动在 `saved_result/{dataset}/0.{score}_epoch{N}/` 下查找预测文件，无需手动重命名。

## 7. 复现步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练 + 测试 dataset2（3 个 epoch，自动保存每个 epoch 的预测）
# <数据目录> 为 A 榜数据集 data_A 的路径
python code/main_tncn.py --dataset dataset2 --data_dir <数据目录> --epochs 3

# 3. 训练 + 测试 dataset1（3 个 epoch）
python code/main_tncn.py --dataset dataset1 --data_dir <数据目录> --epochs 3

# 4. 排名集成（main_tncn 已自动把每个 epoch 的预测保存到
#    saved_result/{dataset}/0.{score}_epoch{N}/{dataset}_result.csv）
# 将第 1、第 3 个 epoch 目录名中的分数（去掉开头的 0.）作为 score1、score2 传入，自动查找：
# 例如 dataset1 的 0.9401_epoch1 与 0.9436_epoch3：
python code/ensemble.py --dataset dataset1 --score1 9401 --score2 9436
# 例如 dataset2 的 0.5924_epoch1 与 0.6103_epoch3：
python code/ensemble.py --dataset dataset2 --score1 5924 --score2 6103
```

输出格式：每行对应一个测试样本，每列对应一个候选商品的得分，无表头和列标签。

## 8. 实验环境

- 操作系统：Ubuntu 22.04
- CUDA 版本：12.4
- Python 版本：3.10
- Jittor 版本：1.3.10
- 深度学习库：Jittor + jittor_geometric（平台预装）
- 其他依赖：numpy, scikit-learn, tqdm, scipy, fire（见 requirements.txt）

## 9. 运行环境要求（已验证）

本代码已在全新 conda 环境中完整验证通过（jittor 1.3.10.0 + jittor_geometric 2.0.0，模型构建、CUDA 前向、loss 计算、反向传播与优化器更新均正常）。运行所需的系统级与 Python 级依赖如下：

### 9.1 系统级依赖

- Ubuntu 22.04，内核 6.8+
- CUDA 12.4（平台提供）
- NVIDIA 驱动版本 >= 12.2（Jittor 据此自动选用 cuda12.2_cudnn8 工具链）
- g++-12（Jittor 的 `cc_path`，例如 `apt install g++-12`）
- 首次 import jittor 时需联网，Jittor 会自动下载约 6GB 的 jtcuda 工具链（CUDA 12.2 + cuDNN 8）到 `~/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/`

### 9.2 Python 级依赖（requirements.txt）

- `jittor==1.3.10.0`（平台预装）
- `jittor_geometric==2.0.0`（平台预装，不在 PyPI，需源码安装：`pip install https://github.com/JittorGeometric/JittorGeometric/archive/refs/heads/master.zip`）
- `numpy<2.0`
- `scipy, scikit-learn, tqdm, pandas, networkx`
- `pymetis`（jittor_geometric 的 partition 模块必需）
- `huggingface_hub`（jittor_geometric 的 datasets 导入链必需）

### 9.3 环境变量（Jittor 工具链）

`cc_path=/usr/bin/g++-12`；`nvcc_path=~/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin/nvcc`；`LD_LIBRARY_PATH` 需包含 CUDA/cuDNN 运行库目录（若已安装 jtcuda，Jittor 的 check_cuda_env 会自动调整 PATH 与 LD_LIBRARY_PATH）。

## 10. 致谢

感谢 jittor_geometric 与官方基线 CRAFT 提供的开源组件与参考实现。