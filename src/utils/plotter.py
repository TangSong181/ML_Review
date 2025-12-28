import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from .config import IMAGE_DIR

def plot_training_history(train_losses, test_losses, test_accuracies, title="Training History", filename="history.png"):
    """
    绘制训练 Loss 和测试 Accuracy 曲线
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss (左轴)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, label='Train Loss', color=color, linestyle='-', marker='o', markersize=4)
    if test_losses:
        ax1.plot(test_losses, label='Test Loss', color='tab:orange', linestyle='--', marker='x')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # 绘制 Accuracy (右轴)
    if test_accuracies:
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy (%)', color=color)  
        ax2.plot(test_accuracies, label='Test Accuracy', color=color, linestyle='-', marker='s', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

    plt.title(title)
    plt.tight_layout()
    
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path)
    print(f"[Plotter] Plot saved to {save_path}")
    plt.close() # 关闭图像防止内存泄漏

def show_image_grid(images, labels, classes, filename="prediction_grid.png"):
    """
    显示图片网格 (用于 CV 预测展示)
    images: Tensor (B, C, H, W) 或 list of numpy arrays
    """
    fig = plt.figure(figsize=(12, 6))
    count = min(len(images), 8) # 最多显示 8 张
    
    for i in range(count):
        ax = plt.subplot(2, 4, i + 1)
        img = images[i]
        
        # 如果是 Tensor，转为 Numpy 并反归一化 (简单假设 mean=0.5, std=0.5 用于显示)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose((1, 2, 0)) # CHW -> HWC
            img = img / 2 + 0.5 # Unnormalize
            img = np.clip(img, 0, 1)
            
        plt.imshow(img, cmap='gray' if img.ndim==2 or img.shape[2]==1 else None)
        
        label_text = classes[labels[i]] if classes else str(labels[i])
        plt.title(f"Pred: {label_text}")
        plt.axis('off')
        
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path)
    print(f"[Plotter] Image grid saved to {save_path}")
    plt.close()
