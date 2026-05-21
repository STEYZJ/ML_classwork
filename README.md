# ML_classwork

2026 春季研究生《机器学习》课程作业仓库。仓库已按课程实验和大作业整理，重点保留可复现源码、实验报告、论文、运行环境配置、结果表和图像。

远程仓库：

```bash
https://github.com/STEYZJ/ML_classwork.git
```

## 目录结构

```text
.
├── dataset/
│   └── dataset/                         # 大作业给定 .mat 数据集
├── 实验/                                # 四次实验原始要求与实验数据
│   ├── 实验一线性回归/
│   ├── LDA实验/
│   ├── SVM 支持向量机人脸识别/
│   └── 第四次实验-神经网络/
├── 大作业/                              # 大作业视频/笔记/要求整理，本地保留大视频
├── 结果/
│   ├── 实验报告/                        # 四次实验报告、源码和输出
│   └── 大作业/                          # 大作业论文、源码和输出
└── README.md
```

说明：`书/`、大作业视频、压缩包、网页资源缓存和系统临时文件不进入版本管理，避免仓库体积过大。

## 四次实验

实验成果位于：

```text
结果/实验报告/
├── 实验报告.md
├── 实验报告.docx
└── 源码/
```

已完成的实验：

1. 线性回归/岭回归；
2. Fisher 线性判别分析 LDA；
3. SVM 支持向量机人脸识别；
4. 全连接神经网络分类器。

一键运行：

```bash
cd 结果/实验报告/源码
conda env create -f environment.yml
conda activate ml-labs

python run_all_experiments.py \
  --experiment-root /Volumes/Work/学习/作业/机器学习/实验 \
  --output-dir outputs
```

本机 Anaconda 运行方式：

```bash
/Volumes/Work/opt/anaconda3/bin/python3 run_all_experiments.py \
  --experiment-root /Volumes/Work/学习/作业/机器学习/实验 \
  --output-dir outputs
```

主要输出：

- `outputs/linear_regression_metrics.csv`
- `outputs/lda_metrics.csv`
- `outputs/lda_sonar_feature_curve.csv`
- `outputs/svm_face_metrics.csv`
- `outputs/neural_network_metrics.csv`
- `outputs/figures/`

当前主要结果：

| 实验 | 数据集/任务 | 主要指标 |
|---|---|---:|
| 线性/岭回归 | sklearn Diabetes | 最佳 R2 = 0.4781 |
| LDA | Iris | ACC = 0.9800 |
| LDA | Sonar | 10 折 ACC = 0.7600 |
| SVM | Yale 人脸识别 | 测试 ACC = 0.7400 |
| 神经网络 | MNIST | 测试 ACC = 0.9033 |
| 神经网络 | Yale | 测试 ACC = 0.5600 |
| 神经网络 | lung | 测试 ACC = 0.8197 |

## 大作业

大作业成果位于：

```text
结果/大作业/
├── 论文.md
├── 论文.docx
└── 源码/
```

题目：基于回归学习与流形结构保持的无监督特征选择。

运行方式：

```bash
cd 结果/大作业/源码
conda env create -f environment.yml
conda activate ml-feature-selection

python run_experiment.py \
  --dataset-dir /Volumes/Work/学习/作业/机器学习/dataset/dataset \
  --output-dir outputs \
  --feature-counts 20 50 100 \
  --knn 5 \
  --alpha 0.01 \
  --beta 0.1 \
  --max-iter 12
```

主要输出：

- `outputs/results.csv`
- `outputs/dataset_summary.csv`
- `outputs/selected_feature_rankings.json`
- `outputs/acc_by_dataset.png`

## 环境说明

两个源码目录均提供独立 Conda 环境文件：

- 四次实验：`结果/实验报告/源码/environment.yml`，环境名 `ml-labs`
- 大作业：`结果/大作业/源码/environment.yml`，环境名 `ml-feature-selection`

也可以使用 pip 安装：

```bash
pip install -r requirements.txt
```

## Git 管理

当前远程仓库：

```bash
git remote -v
# origin  https://github.com/STEYZJ/ML_classwork.git
```

推荐提交流程：

```bash
git status
git add README.md .gitignore dataset 实验 大作业 结果
git commit -m "Update lab reports and project README"
git push origin main
```

`.gitignore` 已忽略：

- macOS 元数据 `.DS_Store`
- Python 缓存 `__pycache__/`、`*.pyc`
- Matplotlib/Pytest 缓存
- 课程参考书目录 `书/`
- 大作业视频
- 压缩包和网页离线资源缓存
