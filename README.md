# ML Review Project

这是一个用于复习和巩固 **机器学习** 与 **深度学习** 基础的实战项目。项目包含从 Python 基础数据处理 (Numpy/Pandas) 到现代深度学习架构 (CNN, ResNet, Transformer) 的完整代码示例。

## 📂 目录结构

```text
ML_Review/
├── data/           # 数据集存放目录 (MNIST, CIFAR-10 等会自动下载至此)
├── docs/           # 学习计划与教程文档
├── outputs/        # 运行产物
│   ├── images/     # 可视化结果 (Loss 曲线, 预测图)
│   ├── logs/       # 运行日志
│   └── models/     # 训练好的模型权重 (.pth)
├── src/            # 源代码
│   ├── basics/     # Python, Numpy, Pandas 基础
│   ├── cv/         # 计算机视觉 (OpenCV, MLP, CNN, Transfer Learning)
│   ├── nlp/        # 自然语言处理 (LSTM, Transformer)
│   └── utils/      # 通用工具 (Trainer, Plotter, Config)
└── requirements.txt # 项目依赖
```

## 🚀 快速开始

### 1. 环境准备

建议使用 Python 3.8+ 环境。

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 运行代码

所有脚本均设计为可独立运行。请在 **项目根目录** (`ML_Review/` 或包含 `src/` 的目录) 下执行命令，以确保 Python 路径解析正确。

**基础示例:**
```bash
python src/basics/02_numpy_arrays.py
python src/basics/03_pandas_dataframes.py
```

**计算机视觉 (CV):**
```bash
# 训练 MNIST MLP
python src/cv/02_mlp_mnist.py

# 训练 CIFAR-10 CNN
python src/cv/03_cnn_cifar10.py
```

**自然语言处理 (NLP):**
```bash
# 训练 LSTM 情感分类
python src/nlp/01_lstm_sentiment.py

# 训练 Transformer
python src/nlp/03_transformer_translator.py
```

## 📊 查看结果

运行结束后，请前往 `outputs/` 目录查看：
*   **`outputs/images/`**: 生成的训练 Loss 曲线图和模型预测结果图。
*   **`outputs/models/`**: 训练好的 PyTorch 模型权重文件。

## 📝 学习计划

详细的学习路线图请参考 [docs/Study_Plan.md](docs/Study_Plan.md)。
