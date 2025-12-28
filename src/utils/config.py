import os
import torch
import numpy as np
import random

# --- 路径配置 ---
# 获取项目根目录 (假设当前文件在 src/utils/config.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 确保目录存在
for d in [DATA_DIR, MODEL_DIR, IMAGE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    return DEVICE

# --- 随机种子 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 保证实验可复现 (会牺牲一点性能)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Config] Random seed set to {seed}")
