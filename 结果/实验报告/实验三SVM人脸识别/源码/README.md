# 实验三 SVM 人脸识别源码说明

## 运行环境

```bash
conda env create -f environment.yml
conda activate ml-labs
```

或使用 pip：

```bash
pip install -r requirements.txt
```

## 运行方式

```bash
python run_experiment.py \
  --experiment-root /Volumes/Work/学习/作业/机器学习/实验 \
  --output-dir outputs
```

本机 Anaconda 验证命令：

```bash
/Volumes/Work/opt/anaconda3/bin/python3 run_experiment.py \
  --experiment-root /Volumes/Work/学习/作业/机器学习/实验 \
  --output-dir outputs
```

## 文件说明

- `run_experiment.py`：实验三独立运行入口；
- `src/svm_face_lab.py`：PCA + RBF SVM 人脸识别实现；
- `src/common.py`：`.mat` 数据读取、混淆矩阵绘制等公共工具；
- `outputs/svm_face_metrics.csv`：SVM 指标；
- `outputs/svm_face_classification_report.txt`：分类报告；
- `outputs/figures/`：混淆矩阵和预测样例图。
