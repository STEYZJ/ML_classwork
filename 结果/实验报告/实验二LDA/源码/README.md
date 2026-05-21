# 实验二 LDA 源码说明

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

- `run_experiment.py`：实验二独立运行入口；
- `src/lda_lab.py`：LDA 与自定义 Fisher LDA 实现；
- `src/common.py`：`.mat` 数据读取、输出目录、CSV 保存等公共工具；
- `outputs/lda_metrics.csv`：Iris 和 Sonar 分类指标；
- `outputs/lda_sonar_feature_curve.csv`：Sonar 特征数量曲线数据；
- `outputs/figures/`：LDA 投影图和特征数量曲线图。
