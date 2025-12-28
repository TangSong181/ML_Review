"""
02_mlp_mnist.py - 使用多层感知机 (MLP) 识别手写数字

【什么是 MNIST?】
MNIST 是深度学习界的“Hello World”。它包含 60,000 张训练图片和 10,000 张测试图片。
每张图片是 28x28 像素的灰度图，也就是一个 0-9 的手写数字。

【什么是 MLP (Multi-Layer Perceptron)?】
最简单的神经网络形式。它由输入层、若干隐藏层和输出层组成。
每一层都与下一层“全连接” (Fully Connected)，即每个神经元都连接到下一层的所有神经元。
"""

import sys
import os
# 将项目根目录添加到 python 路径，以便导入 src.utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.utils import Trainer, set_seed, DATA_DIR, MODEL_DIR, plot_training_history, show_image_grid

# 1. 设置随机种子，保证实验结果可复现
set_seed(42)

# 2. 数据准备
def get_data_loaders(batch_size=64):
    # 定义预处理管道
    transform = transforms.Compose([
        transforms.ToTensor(), # 将图片 (0-255) 转换为 Tensor (0.0-1.0), 且形状从 HWC 变为 CHW
        transforms.Normalize((0.1307,), (0.3081,)) # 标准化: (input - mean) / std。这是 MNIST 的统计均值和方差。
    ])
    
    # 下载并加载数据集
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    # DataLoader 负责分批次 (Batch) 读取数据，shuffle=True 表示每轮训练打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# 3. 模型定义
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Flatten: 将 28x28 的二维图像展平成 784 维的一维向量
        self.flatten = nn.Flatten()
        
        # Sequential: 一个有序的容器，数据会按顺序通过这些层
        self.fc_layers = nn.Sequential(
            # 第一层: 输入 784 -> 输出 512
            nn.Linear(784, 512),
            # ReLU (Rectified Linear Unit): 激活函数，f(x) = max(0, x)。引入非线性，让神经网络能拟合复杂函数。
            nn.ReLU(),
            
            # 第二层: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            
            # 输出层: 256 -> 10 (对应 0-9 十个数字的评分/Logits)
            nn.Linear(256, 10)
        )

    def forward(self, x):
        """前向传播逻辑"""
        # x.shape: [Batch_Size, 1, 28, 28]
        x = self.flatten(x) 
        # x.shape: [Batch_Size, 784]
        logits = self.fc_layers(x)
        return logits

if __name__ == "__main__":
    # 超参数配置
    BATCH_SIZE = 64
    EPOCHS = 3      # 训练轮数
    LR = 0.01       # 学习率 (Learning Rate)

    # 初始化
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    model = SimpleMLP()
    
    # 定义优化器: SGD (随机梯度下降)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # 定义损失函数: 交叉熵损失 (CrossEntropyLoss)，用于多分类任务
    criterion = nn.CrossEntropyLoss()

    # 使用我们封装好的 Trainer 进行训练
    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader)
    history = trainer.train(EPOCHS)

    # 绘图并保存
    plot_training_history(history['train_loss'], history['test_loss'], history['test_acc'], 
                          filename="mnist_history.png")

    # 保存训练好的模型权重
    model_path = os.path.join(MODEL_DIR, "mnist_mlp.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 预测可视化 (取一个 Batch 看看效果)
    data, target = next(iter(test_loader))
    outputs = model.to("cpu")(data) # 简单演示，用 CPU 推理
    _, predicted = torch.max(outputs, 1) # 获取概率最大的类别索引
    show_image_grid(data, predicted, classes=None, filename="mnist_predictions.png")