"""
02_transformer_mechanism.py - 手动实现 Self-Attention 机制

【目标】
Transformer 是现代 NLP (如 ChatGPT) 的基石。
它的核心不是 RNN 的循环，而是 "Attention" (注意力机制)。
本脚本不进行训练，而是手动实现并可视化 Self-Attention 的计算过程。

【核心公式】
Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from src.utils import get_device, IMAGE_DIR

device = get_device()
print(f"Using device: {device}")

# 2. Positional Encoding (位置编码)
# 因为 Transformer 并行处理所有词，它不知道 "I love" 和 "love I" 的区别。
# 必须显式地把位置信息加到 Embedding 里。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 使用不同频率的 sin/cos 函数生成位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x + pe (将位置向量直接加到词向量上)
        x = x + self.pe[:, :x.size(1), :]
        return x

# 3. 手动实现 Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        
        # 定义三个线性变换矩阵 Wq, Wk, Wv
        # 它们把输入向量映射到 Query, Key, Value 空间
        self.query = nn.Linear(embed_dim, head_dim)
        self.key   = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        
        # 1. 生成 Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. 计算注意力分数 (Scaled Dot-Product)
        # Q @ K.T: 计算每个 Query 和所有 Key 的相似度
        # / sqrt(d): 缩放，防止数值过大导致 softmax 梯度消失
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. 归一化 (Softmax)
        # 得到概率分布，和为 1
        attention_weights = F.softmax(scores, dim=-1)

        # 4. 加权求和
        # 用权重去加权 Value
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

def visualize_attention(attention_weights, tokens, output_path):
    """绘制注意力热力图"""
    weights = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(weights, cmap='viridis')
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", color="w")

    plt.title("Self-Attention Map")
    plt.xlabel("Key (Attention To)")
    plt.ylabel("Query (Current Word)")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Attention Map saved to '{output_path}'")

if __name__ == "__main__":
    print("--- Manual Self-Attention Demo ---")
    tokens = ["I", "love", "machine", "learning"]
    seq_len = len(tokens)
    embed_dim = 4  
    head_dim = 4   

    inputs = torch.randn(1, seq_len, embed_dim).to(device)

    # 1. 添加位置编码
    pe_layer = PositionalEncoding(d_model=embed_dim).to(device)
    inputs_with_pe = pe_layer(inputs)

    # 2. 计算 Attention
    attention_layer = SelfAttention(embed_dim, head_dim).to(device)
    output, weights = attention_layer(inputs_with_pe)
    
    # 3. 可视化
    output_img_path = os.path.join(IMAGE_DIR, "attention_map.png")
    visualize_attention(weights[0], tokens, output_img_path)