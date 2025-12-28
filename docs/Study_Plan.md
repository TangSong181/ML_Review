# 机器学习与图像处理复习计划

这个计划旨在帮助你快速回顾 Python 数据科学栈，并深入复习深度学习的核心概念（CNN, RNN, Transformer）。我们将使用 **PyTorch** 作为主要的深度学习框架，并结合 **OpenCV** 进行图像处理。

## 零阶段：Python 基础 (预计 0.5 天)
**目标**：巩固 Python 编程基础。

### 0. Python 核心概念 [x] 已完成
*   **变量与数据类型**: 列表、元组、字典、集合。
*   **控制流**: 条件语句、循环。
*   **函数**: 定义、参数、返回值。
*   **类与对象**: 基本面向对象编程。
*   **代码实例**: `src/basics/01_python_syntax.py` (Python 语法练习，含异常处理与文件操作)

---

## 第一阶段：Python 数据科学与图像处理基础 (预计 1-2 天)
**目标**：重温数据操作与传统图像处理算法。

### 1. Python 数据科学栈 [x] 已完成
*   **NumPy**: 数组操作、广播机制、矩阵运算。
*   **Matplotlib / Seaborn**: 数据可视化。
*   **代码实例**: `src/basics/02_numpy_arrays.py` (矩阵运算实战)

### 1.5. Pandas 数据处理 [x] 已完成
*   **DataFrame**: 数据读取 (read_csv)、筛选、排序。
*   **数据清洗**: 处理缺失值 (fillna), 数据合并 (merge/concat)。
*   **统计分析**: groupby, pivot_table.
*   **代码实例**: `src/basics/03_pandas_dataframes.py` (数据分析实战)

### 2. OpenCV 图像处理 [x] 已完成
*   **基本操作**: 读取、缩放、旋转、色彩空间转换 (RGB <-> HSV/GRAY)。
*   **核心算法**: 边缘检测 (Canny), 图像平滑 (Gaussian Blur), 阈值处理。
*   **代码实例**: `src/cv/01_opencv_pipeline.py` (图像预处理流水线)

---

## 第二阶段：深度学习基础与 CNN (预计 2-3 天)
**目标**：理解神经网络核心并掌握计算机视觉基石。

### 3. PyTorch 基础与 MLP
*   **Tensor**: 自动求导 (Autograd)。
*   **模型构建**: `nn.Module`, 损失函数, 优化器 (SGD, Adam)。
*   **代码实例**: `src/cv/02_mlp_mnist.py` (全连接网络识别手写数字)

### 4. 卷积神经网络 (CNN)
*   **理论**: 卷积层、池化层、感受野、Padding/Stride。
*   **经典架构**: 复习 LeNet, AlexNet, ResNet (残差连接)。
*   **实战优化**: 使用 `tqdm` 进度条, `sklearn` 分类报告, 模型保存与加载。
*   **代码实例**: `src/cv/03_cnn_cifar10.py` (搭建一个简单的 CNN 进行图片分类)

### 4.5. 迁移学习 (Transfer Learning) (新增)
*   **理论**: 预训练模型 (Pre-trained Models), 特征提取 (Feature Extraction) vs 微调 (Fine-tuning)。
*   **代码实例**: `src/cv/04_resnet_transfer.py` (使用预训练的 ResNet18 处理 CIFAR-10)

---

## 第三阶段：序列模型 (RNN) (预计 2 天)
**目标**：处理时间序列与文本数据。

### 5. 循环神经网络 (RNN) 及其变体
*   **理论**: RNN 的梯度消失问题，LSTM 和 GRU 的门控机制。
*   **Batch 处理**: 使用 `DataLoader`, `Dataset` 和 `pad_sequence` 处理变长序列。
*   **代码实例**: `src/nlp/01_lstm_sentiment.py` (使用 LSTM 进行简单文本情感分类)

---

## 第四阶段：Attention 与 Transformer (预计 3-4 天)
**目标**：掌握现代深度学习的核心架构。

### 6. Attention 机制与 Transformer
*   **理论**: Self-Attention (自注意力), Multi-Head Attention, Positional Encoding (位置编码)。
*   **架构**: Encoder-Decoder 结构 (这是 BERT 和 GPT 的基础)。
*   **代码实例**: 
    *   `src/nlp/02_transformer_mechanism.py` (实现并可视化 Self-Attention 模块)
    *   `src/nlp/03_transformer_translator.py` (Seq2Seq Transformer 字符串翻转任务)

---

## 下一步行动建议
1.  安装依赖: `pip install -r requirements.txt`
2.  按照上述顺序运行 `src/` 下的代码。
3.  在 `outputs/` 目录查看运行结果（图片、模型、日志）。
