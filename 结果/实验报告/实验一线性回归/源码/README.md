# 实验一线性回归源码说明

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
python run_experiment.py --output-dir outputs
```

本机 Anaconda 验证命令：

```bash
/Volumes/Work/opt/anaconda3/bin/python3 run_experiment.py --output-dir outputs
```

## 文件说明

- `run_experiment.py`：实验一独立运行入口；
- `src/linear_regression_lab.py`：线性回归与岭回归实现；
- `src/common.py`：输出目录、CSV 保存等公共工具；
- `outputs/linear_regression_metrics.csv`：实验指标；
- `outputs/figures/`：预测散点图和岭回归系数路径图。
