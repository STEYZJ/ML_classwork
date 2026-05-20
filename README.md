# ML_classwork

2026 春季研究生《机器学习》课程作业仓库。本仓库整理了课程资料、给定数据集，以及大作业“基于回归学习及流形结构保持的无监督特征选择”的报告、论文、源码和实验结果。

## 目录结构

```text
.
├── dataset/dataset/                  # 大作业给定 .mat 数据集
│   ├── JAFFE.mat
│   ├── MNIST.mat
│   ├── TOX_171.mat
│   ├── Yale.mat
│   └── lung.mat
├── 结果/
│   ├── 大作业要求整理.md / .docx
│   ├── 实验报告/
│   │   ├── 实验报告.md / .docx
│   │   └── 源码/
│   └── 大作业/
│       ├── 论文.md / .docx
│       └── 源码/
└── 基于回归学习以及流形结构保持的无监督特征选择.md
```

## 交付内容

- `结果/大作业要求整理.md`：根据视频与作业要求整理的任务说明。
- `结果/实验报告/实验报告.md`：实验流程、数据集、方法、实验设置与结果分析。
- `结果/大作业/论文.md`：小论文形式的大作业文本，包含题目、摘要、引言、相关工作、方法、实验和结论。
- `结果/实验报告/源码` 与 `结果/大作业/源码`：两套可独立运行的源码，均包含 README、Conda 环境文件、依赖文件和实验输出。

## 程序环境

推荐使用 Conda 创建环境：

```bash
cd 结果/大作业/源码
conda env create -f environment.yml
conda activate ml-feature-selection
```

也可以用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- Python 3.10
- numpy
- scipy
- scikit-learn
- pandas
- matplotlib

## 运行实验

在 `结果/大作业/源码` 或 `结果/实验报告/源码` 下执行：

```bash
python run_experiment.py \
  --dataset-dir /Volumes/Work/学习/作业/机器学习/dataset/dataset \
  --output-dir outputs \
  --feature-counts 20 50 100 \
  --knn 5 \
  --alpha 0.01 \
  --beta 0.1 \
  --max-iter 12
```

运行后会生成：

- `outputs/results.csv`：不同数据集、方法和特征数下的 ACC、NMI、ARI、F1。
- `outputs/dataset_summary.csv`：数据集规模、图边数、迭代次数和目标函数值。
- `outputs/selected_feature_rankings.json`：特征排序结果。
- `outputs/acc_by_dataset.png`：准确率对比图。

## 方法概要

源码实现了基于回归学习、流形结构保持和 \(L_{2,1}\) 行稀疏约束的无监督特征选择方法。训练阶段仅使用 `fea` 特征矩阵，`gnd` 标签只用于聚类评价。算法流程包括标准化、kNN 图构造、图拉普拉斯矩阵计算、谱嵌入伪标签生成、稀疏图正则回归求解和按回归矩阵行范数排序选特征。

## Git 版本管理

本仓库远程地址：

```bash
git remote -v
# origin  https://github.com/STEYZJ/ML_classwork.git
```

推荐日常提交流程：

```bash
git status
git add README.md .gitignore 结果
git commit -m "Update assignment docs and experiment source"
git push origin main
```

`.gitignore` 已忽略 macOS 元数据、Python 缓存、Matplotlib 缓存和体积较大的原始视频/部分参考资料，避免把临时文件继续提交到仓库。
