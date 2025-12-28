"""
03_cnn_cifar10.py - 使用卷积神经网络 (CNN) 识别 CIFAR-10 图片

【什么是 CIFAR-10?】
一个经典的彩色图片数据集，包含 10 个类别：飞机、汽车、鸟、猫、鹿、狗、蛙、马、船、卡车。
图片大小为 32x32，3 通道 (RGB)。

【什么是 CNN?】
CNN 专门为图像设计。不同于 MLP 将图片展平，CNN 保留了图片的二维空间结构。
- 卷积层 (Conv): 提取局部特征 (如边缘、纹理)。
- 池化层 (Pool): 降低特征图尺寸，减少计算量，提取主要特征。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from src.utils import Trainer, set_seed, DATA_DIR, MODEL_DIR, plot_training_history, show_image_grid

set_seed(42)

# CIFAR-10 的类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_loaders(batch_size=64):
    # 训练集数据增强: 随机翻转，让模型见过更多样的数据，防止过拟合
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化到 [-1, 1]
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    # num_workers=0 在 Windows 上通常为了避免多进程报错
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1: 输入 3 通道 (RGB), 输出 32 个特征图 (Feature Maps), 卷积核大小 3x3
        # padding=1 保证卷积后尺寸不变 (32x32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        
        # 最大池化层: 2x2 区域取最大值，图片尺寸减半 (32->16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层 2: 输入 32 通道, 输出 64 通道
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # 全连接层
        # 图片经过两次池化 (32 -> 16 -> 8)，最终尺寸是 8x8
        # 输入维度: 64 (通道数) * 8 * 8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10) # 输出 10 类

    def forward(self, x):
        # x: [Batch, 3, 32, 32]
        
        # 第一层: Conv -> ReLU -> Pool
        # Out: [Batch, 32, 16, 16]
        x = self.pool(torch.relu(self.conv1(x)))
        
        # 第二层: Conv -> ReLU -> Pool
        # Out: [Batch, 64, 8, 8]
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 展平: [Batch, 4096]
        x = x.view(-1, 64 * 8 * 8)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 0.001

    trainloader, testloader = get_loaders(BATCH_SIZE)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    # 使用 Adam 优化器，通常比 SGD 收敛更快
    optimizer = optim.Adam(model.parameters(), lr=LR)

    trainer = Trainer(model, optimizer, criterion, trainloader, testloader)
    history = trainer.train(EPOCHS)
    
    plot_training_history(history['train_loss'], history['test_loss'], history['test_acc'], filename="cifar10_history.png")
    
    model_path = os.path.join(MODEL_DIR, "cifar10_cnn.pth")
    torch.save(model.state_dict(), model_path)
    
    # 预测演示
    images, labels = next(iter(testloader))
    outputs = model.to("cpu")(images)
    _, predicted = torch.max(outputs, 1)
    show_image_grid(images, predicted, classes, filename="cifar10_predictions.png")