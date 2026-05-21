# ML_classwork

2026 春季研究生《机器学习》课程作业仓库。仓库已按课程实验和大作业整理，重点保留可复现源码、实验报告、论文、运行环境配置、结果表和图像。

远程仓库：

```bash
https://github.com/STEYZJ/ML_classwork.git
```

<!-- repo-meta:start -->
- 当前版本：`2026.05.21.3`
- 稳定分支：`main`
- 远程仓库：`https://github.com/STEYZJ/ML_classwork`
- GitHub About：2026春季研究生《机器学习》课程作业：四次实验、无监督特征选择大作业、报告与可复现实验源码
- Topics：`machine-learning` `coursework` `feature-selection` `regression` `lda` `svm` `neural-network` `python`
<!-- repo-meta:end -->

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
├── README.md
├── 实验一线性回归/
├── 实验二LDA/
├── 实验三SVM人脸识别/
└── 实验四神经网络/
```

已完成的实验：

1. 线性回归/岭回归；
2. Fisher 线性判别分析 LDA；
3. SVM 支持向量机人脸识别；
4. 全连接神经网络分类器。

每次实验均拆分为独立目录，目录内包含 `实验报告.md`、`实验报告.docx`、`源码/README.md`、独立运行脚本和对应输出结果。

实验一运行：

```bash
cd 结果/实验报告/实验一线性回归/源码
conda env create -f environment.yml
conda activate ml-labs

python run_experiment.py --output-dir outputs
```

实验二、三、四运行方式类似，以实验二为例：

```bash
cd 结果/实验报告/实验二LDA/源码
python run_experiment.py \
  --experiment-root /Volumes/Work/学习/作业/机器学习/实验 \
  --output-dir outputs
```

主要输出：

- `结果/实验报告/实验一线性回归/源码/outputs/linear_regression_metrics.csv`
- `结果/实验报告/实验二LDA/源码/outputs/lda_metrics.csv`
- `结果/实验报告/实验二LDA/源码/outputs/lda_sonar_feature_curve.csv`
- `结果/实验报告/实验三SVM人脸识别/源码/outputs/svm_face_metrics.csv`
- `结果/实验报告/实验四神经网络/源码/outputs/neural_network_metrics.csv`

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

实验和大作业源码目录均提供独立 Conda 环境文件：

- 四次实验：`结果/实验报告/实验*/源码/environment.yml`，环境名 `ml-labs`
- 大作业：`结果/大作业/源码/environment.yml`，环境名 `ml-feature-selection`

也可以使用 pip 安装：

```bash
pip install -r requirements.txt
```

## 自动化管理

当前远程仓库：

```bash
git remote -v
# origin  https://github.com/STEYZJ/ML_classwork.git
```

仓库已经提供自动管理配置：

- `.github/repo-meta.json`：统一维护仓库 About、Topics、默认分支、允许分支前缀和当前版本；
- `VERSION`：当前版本号，GitHub Actions 会在 `main` 分支按该文件自动创建版本标签；
- `scripts/repo_manage.py`：本地管理脚本，可检查元数据、更新 README、升级版本、创建规范分支、同步 GitHub About；
- `.github/workflows/repo-check.yml`：每次 push/PR 自动检查版本、README 元信息和分支命名；
- `.github/workflows/version-tag.yml`：`VERSION` 变化后自动创建并推送 `v版本号` 标签；
- `.github/workflows/sync-about.yml`：配置仓库 Secret `REPO_ADMIN_TOKEN` 后自动同步 GitHub About 和 Topics。

常用命令：

```bash
# 检查版本、README 元数据和当前分支命名
python scripts/repo_manage.py check

# 根据 .github/repo-meta.json 自动更新 VERSION 和 README 元数据块
python scripts/repo_manage.py update

# 按日期升级版本，例如 2026.05.21 -> 2026.05.21.1
python scripts/repo_manage.py bump

# 创建规范分支，可选 work/docs/experiment/fix/release
python scripts/repo_manage.py branch docs "update readme" --push

# 使用 REPO_ADMIN_TOKEN 或 GH_TOKEN 同步 GitHub About
python scripts/repo_manage.py sync-about

# 根据 VERSION 创建并推送版本标签
python scripts/repo_manage.py tag --push
```

`.gitignore` 已忽略：

- macOS 元数据 `.DS_Store`
- Python 缓存 `__pycache__/`、`*.pyc`
- Matplotlib/Pytest 缓存
- 课程参考书目录 `书/`
- 大作业视频
- 压缩包和网页离线资源缓存
