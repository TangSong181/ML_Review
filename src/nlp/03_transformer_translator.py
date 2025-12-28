"""
03_transformer_translator.py - Transformer 训练实战 (Seq2Seq)

【任务】
训练一个 Transformer 模型，学会将字符串“翻转”。
输入: "a b c" -> 输出: "c b a"
虽然任务简单，但它涵盖了机器翻译的所有核心技术。

【关键技术点】
1. **Seq2Seq (序列到序列)**: 输入和输出都是序列，且长度可能不同。
2. **Teacher Forcing**: 训练时，Decoder 的输入是真实的标签 (Shifted Right)，而不是上一步的预测值。
3. **Masking (掩码)**:
   - **Padding Mask**: 忽略填充的 0。
   - **Look-ahead Mask (Causal Mask)**: 防止 Decoder 在预测第 t 个词时“偷看”到 t+1 之后的词。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.utils import Trainer, set_seed, get_device, plot_training_history

set_seed(42)
device = get_device()

# ... (数据准备部分代码保持不变，只需少量注释) ...
# [省略部分辅助函数，重点注释模型类]

vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
    vocab[char] = i + 3
idx_to_char = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)

def generate_random_string(length=10):
    length = random.randint(3, length)
    return [random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length)]

def prepare_data(num_samples=1000, max_len=10):
    data = []
    for _ in range(num_samples):
        seq_chars = generate_random_string(max_len)
        src = [vocab[c] for c in seq_chars]
        tgt = [vocab[c] for c in reversed(seq_chars)]
        
        # Decoder 的输入需要 <SOS> (Start of Sentence)
        tgt_in = [vocab['<SOS>']] + tgt
        # Decoder 的目标输出是对应的字符 + <EOS> (End of Sentence)
        tgt_out = tgt + [vocab['<EOS>']]
        
        data.append((torch.tensor(src), torch.tensor(tgt_in), torch.tensor(tgt_out)))
    return data

class ReverseDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    srcs, tgt_ins, tgt_outs = zip(*batch)
    srcs = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_ins = pad_sequence(tgt_ins, batch_first=True, padding_value=0)
    tgt_outs = pad_sequence(tgt_outs, batch_first=True, padding_value=0)
    return srcs, tgt_ins, tgt_outs

# --- Transformer 模型定义 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerSeq2Seq, self).__init__()
        # Embedding: 将整数索引映射为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # PyTorch 官方 Transformer 模块
        # 包含了 Encoder 和 Decoder 的所有层
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            batch_first=True 
        )
        # 最后的线性层: 将向量映射回词汇表大小，计算每个词的概率
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # src: [Batch, Src_Len] (源序列)
        # tgt: [Batch, Tgt_Len] (目标序列输入)
        
        # 1. Padding Mask (掩码)
        # 标记哪些位置是填充的 0，Transformer 会忽略这些位置
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)
        
        # 2. Causal Mask (因果掩码 / Look-ahead Mask)
        # 这是一个上三角矩阵，用于 Decoder。
        # 确保在预测位置 t 时，只能看到 t 之前的信息，不能看到未来。
        seq_len = tgt.shape[1]
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # 3. Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(64))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(64))
        
        # 4. Transformer Forward
        out = self.transformer(
            src_emb, tgt_emb, 
            tgt_mask=tgt_mask, # 应用因果掩码
            src_key_padding_mask=src_key_padding_mask, # 应用 Encoder Padding 掩码
            tgt_key_padding_mask=tgt_key_padding_mask, # 应用 Decoder Padding 掩码
            memory_key_padding_mask=src_key_padding_mask # Decoder 注意力查 Encoder 输出时的掩码
        )
        return self.fc_out(out)

def train_transformer():
    # 简单的训练循环
    BATCH_SIZE = 32
    EPOCHS = 10
    
    train_data = prepare_data(2000)
    train_loader = DataLoader(ReverseDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = TransformerSeq2Seq(VOCAB_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 计算 Loss 时忽略 Padding (索引0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print("Starting Transformer Training (String Reversal Task)...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt_in)
            
            # Reshape: CrossEntropyLoss 需要 (N, C) 格式
            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    plot_training_history(train_losses, [], [], filename="transformer_loss.png")
    return model

def inference(model, src_str):
    """推理逻辑: 贪心解码 (Greedy Decoding)"""
    model.eval()
    src = [vocab[c] for c in src_str]
    src_tensor = torch.tensor(src).unsqueeze(0).to(device)
    
    # 初始 Decoder 输入: 只有 <SOS>
    tgt_indices = [vocab['<SOS>']]
    
    print(f"Input: '{src_str}'", end=" -> ")
    
    # 循环生成，直到生成 <EOS> 或达到最大长度
    for i in range(len(src_str) + 5):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        
        # 获取最后一个时间步的预测结果
        next_token_logits = output[0, -1, :]
        next_token_id = next_token_logits.argmax().item()
        
        if next_token_id == vocab['<EOS>']:
            break
            
        tgt_indices.append(next_token_id)
        
    result = "".join([idx_to_char.get(idx, '') for idx in tgt_indices[1:]])
    print(f"Pred: '{result}'")

if __name__ == "__main__":
    trained_model = train_transformer()
    print("\n--- Testing ---")
    inference(trained_model, "hello")
    inference(trained_model, "pytorch")