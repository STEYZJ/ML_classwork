# 四次实验源码说明

## 实验内容

本目录包含 4 次机器学习实验的统一源码：

1. 实验一：线性回归/岭回归；
2. 实验二：Fisher 线性判别分析 LDA；
3. 实验三：SVM 支持向量机人脸识别；
4. 实验四：全连接神经网络分类器。

## 运行环境

推荐使用 Conda 创建独立环境：

```bash
conda env create -f environment.yml
conda activate ml-labs
```

也可以使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

本机已验证可用的 Python：

```bash
/Volumes/Work/opt/anaconda3/bin/python3
```

## 一键运行

在本目录下执行：

```bash
python run_all_experiments.py \
  --experiment-root /Volumes/Work/学习/作业/机器学习/实验 \
  --output-dir outputs
```

若使用本机 Anaconda：

```bash
/Volumes/Work/opt/anaconda3/bin/python3 run_all_experiments.py \
  --experiment-root /Volumes/Work/学习/作业/机器学习/实验 \
  --output-dir outputs
```

## 输出文件

- `outputs/linear_regression_metrics.csv`：线性回归与岭回归指标。
- `outputs/lda_metrics.csv`：Iris 和 Sonar 上 LDA 分类指标。
- `outputs/lda_sonar_feature_curve.csv`：Sonar 不同特征数下的 LDA 准确率。
- `outputs/svm_face_metrics.csv`：Yale 人脸数据上的 SVM 分类结果。
- `outputs/neural_network_metrics.csv`：MNIST、Yale、lung 上的神经网络分类结果。
- `outputs/figures/`：实验图像，包括预测散点图、准确率曲线、混淆矩阵等。

## 数据说明

- LDA 实验使用 `实验/LDA实验/LDA实验/iris.mat` 和 `sonar.mat`。
- SVM 人脸识别使用本地 Yale 人脸数据 `实验/第四次实验-神经网络/第四次实验-神经网络/datasets/Yale.mat`，避免在线下载 LFW 数据集失败。
- 神经网络实验使用 `MNIST.mat`、`Yale.mat`、`lung.mat`。
- 线性回归实验使用 sklearn 自带 Diabetes 回归数据集，作为无需联网的回归应用问题。
