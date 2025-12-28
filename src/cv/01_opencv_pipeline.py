"""
01_opencv_pipeline.py - OpenCV 图像处理基础流水线

【学习目标】
1. 掌握图像的读取与显示。
2. 理解图像的平滑 (去噪)、边缘检测、阈值处理。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.utils.config import IMAGE_DIR

def create_synthetic_image() -> np.ndarray:
    """
    创建一个包含几何图形和噪点的合成图像，用于测试。
    """
    # 创建一个 400x400 的黑色背景，3通道 (RGB)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 1. 画一个实心圆 (红色)
    cv2.circle(img, (100, 100), 50, (255, 0, 0), -1)
    
    # 2. 画一个实心矩形 (绿色)
    cv2.rectangle(img, (200, 50), (350, 200), (0, 255, 0), -1)
    
    # 3. 画一条线 (蓝色)
    cv2.line(img, (50, 350), (350, 350), (0, 0, 255), 5)
    
    # 4. 添加随机椒盐噪点
    noise = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    noisy_img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
    
    return noisy_img

def process_image(img: np.ndarray):
    """
    OpenCV 处理流水线
    """
    print("正在处理图像...")
    
    # BGR -> RGB & BGR -> Gray
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊 (去噪)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Canny 边缘检测
    edges = cv2.Canny(img_blur, 100, 200)
    
    # Otsu 二值化
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img_rgb, img_gray, img_blur, edges, img_thresh

def show_results(original, gray, blur, edges, threshold):
    """
    显示并保存结果
    """
    plt.figure(figsize=(15, 10))
    
    titles = ["Original (with Noise)", "Grayscale", "Gaussian Blur (Denoised)", "Canny Edges", "Otsu Thresholding"]
    images = [original, gray, blur, edges, threshold]
    cmaps = [None, 'gray', 'gray', 'gray', 'gray']
    
    print(f"处理后的图片将保存到目录: {IMAGE_DIR}")

    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        plt.subplot(2, 3, i + 1)
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        
        # 保存
        filename = f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(IMAGE_DIR, filename)
        
        if cmap == 'gray':
            cv2.imwrite(filepath, img)
        else:
            # RGB -> BGR for cv2.imwrite
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    # plt.show() # 非阻塞模式不显示窗口
    print("OpenCV 处理完成。")

if __name__ == "__main__":
    # 1. 创建图像
    img = create_synthetic_image()
    
    # 2. 处理
    results = process_image(img)
    
    # 3. 显示与保存
    show_results(*results)