"""
04_resnet_transfer.py - 迁移学习 (Transfer Learning) 实战

【什么是迁移学习?】
不要从头开始训练！
使用在 ImageNet (1000类, 120万张图) 上预训练好的模型 (如 ResNet)，
利用它已经学到的特征提取能力，来解决我们自己的小数据集任务 (如 CIFAR-10)。

【策略】
1. Feature Extraction (特征提取): 冻结前面的卷积层，只训练最后的全连接层。
2. Fine-tuning (微调): 解冻部分或全部层，以极小的学习率进行更新。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from src.utils import Trainer, set_seed, DATA_DIR, MODEL_DIR, plot_training_history, get_device

set_seed(42)
device = get_device()

def get_loaders(batch_size=128):
    # ResNet 期望输入至少 224x224，但为了速度我们这里 Resize 到 64x64
    # 这比 32x32 效果好，但比 224 快得多
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet 均值
    ])

    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

def get_pretrained_model():
    print("Loading pretrained ResNet18...")
    # 加载预训练权重
    model = models.resnet18(pretrained=True)
    
    # 策略: 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后的全连接层 (fc)
    # 新的层默认 requires_grad=True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10) # CIFAR-10 有 10 类
    
    return model

if __name__ == "__main__":
    BATCH_SIZE = 64 # 显存如果不够可以调小
    EPOCHS = 2      # 迁移学习收敛很快，2轮足矣看效果

    train_loader, test_loader = get_loaders(BATCH_SIZE)
    model = get_pretrained_model()
    
    criterion = nn.CrossEntropyLoss()
    # 只优化 fc 层的参数
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 使用通用 Trainer
    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader)
    history = trainer.train(EPOCHS)
    
    # 绘图
    plot_training_history(history['train_loss'], history['test_loss'], history['test_acc'], 
                          filename="transfer_learning_history.png")
    
    # 保存
    model_path = os.path.join(MODEL_DIR, "cifar10_resnet_finetuned.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")