# 机器学习实验报告目录

本目录按 4 次实验拆分存放，每次实验均包含独立实验报告、源码、运行环境文件和实验输出。

## 目录结构

```text
实验报告/
├── 实验一线性回归/
│   ├── 实验报告.md
│   ├── 实验报告.docx
│   └── 源码/
├── 实验二LDA/
│   ├── 实验报告.md
│   ├── 实验报告.docx
│   └── 源码/
├── 实验三SVM人脸识别/
│   ├── 实验报告.md
│   ├── 实验报告.docx
│   └── 源码/
└── 实验四神经网络/
    ├── 实验报告.md
    ├── 实验报告.docx
    └── 源码/
```

每个 `源码/` 目录均包含：

- `README.md`：本实验运行方式；
- `environment.yml`：Conda 环境；
- `requirements.txt`：pip 依赖；
- `run_experiment.py`：本实验独立运行入口；
- `src/`：本实验源码；
- `outputs/`：已生成的指标文件和图像结果。

## 统一环境

4 次实验依赖一致，可任选一个实验源码目录创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate ml-labs
```

也可以使用 pip：

```bash
pip install -r requirements.txt
```
