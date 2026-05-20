# 大作业源码说明

## 运行环境

- 推荐使用 Conda 创建独立环境
- Python 3.10
- 依赖包：`numpy`、`scipy`、`scikit-learn`、`pandas`、`matplotlib`
- 数据集目录：`/Volumes/Work/学习/作业/机器学习/dataset/dataset`

创建并激活 Conda 环境：

```bash
conda env create -f environment.yml
conda activate ml-feature-selection
```

也可以使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

如果不创建新环境，也可以使用本机已验证可用的 Anaconda Python：

```bash
/Volumes/Work/opt/anaconda3/bin/python3 run_experiment.py --dataset-dir /Volumes/Work/学习/作业/机器学习/dataset/dataset --output-dir outputs
```

## 运行方式

在本目录下执行：

```bash
python run_experiment.py --dataset-dir /Volumes/Work/学习/作业/机器学习/dataset/dataset --output-dir outputs
```

可调参数示例：

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

## 输出文件

- `outputs/results.csv`：不同数据集、不同特征选择方法和特征数下的聚类指标。
- `outputs/dataset_summary.csv`：数据集规模、图边数、迭代次数和最终目标函数值。
- `outputs/selected_feature_rankings.json`：每个数据集的前若干个特征排序索引。
- `outputs/acc_by_dataset.png`：准确率结果图。

## 方法说明

源码实现“基于回归学习及流形结构保持的无监督特征选择”。训练阶段只使用特征矩阵 `fea`，不使用标签 `gnd`；标签只用于实验评价。算法先构造 kNN 图和拉普拉斯矩阵，再用谱嵌入形成无监督伪标签，最后求解带 \(L_{2,1}\) 行稀疏约束和流形正则项的回归模型，并按 \(W\) 的行范数对特征排序。
