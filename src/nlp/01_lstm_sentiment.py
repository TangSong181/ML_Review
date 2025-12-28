"""
01_lstm_sentiment.py - 使用 LSTM 进行文本情感分类

【任务目标】
输入一句话 (如 "I love this movie")，判断它是正面 (Positive) 还是负面 (Negative) 情感。
这是 NLP 中最基础的“序列分类”任务。

【难点】
1. **变长序列**: 句子长度不一，需要用 Padding (填充) 对齐。
2. **时序依赖**: "Not bad" 和 "Bad" 意思完全相反，模型需要理解上下文顺序。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.utils import get_device

# 1. 设置设备
device = get_device()
print(f"Using device: {device}")

# 2. 准备数据集 (模拟数据)
# 0: Negative, 1: Positive
raw_data = [
    ("I love this movie", 1),
    ("This film is great", 1),
    ("I like this book", 1),
    ("What a wonderful day", 1),
    ("I hate this bug", 0),
    ("This code is bad", 0),
    ("Terrible experience", 0),
    ("I am sad", 0),
    ("Fantastic performance", 1),
    ("Awful service", 0),
    ("Really really good", 1), 
    ("No", 0)
]

# 构建词汇表 (Vocabulary)
# 将每个单词映射到一个唯一的整数索引
word_to_ix = {"<PAD>": 0, "<UNK>": 1} # 0 用于填充，1 用于未知词
for sentence, tag in raw_data:
    for word in sentence.lower().split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(f"Vocab size: {len(word_to_ix)}")
print(f"Vocab: {word_to_ix}")

# 自定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, data, word_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        # 将单词转换为索引，如果词表中没有，就用 <UNK> (1)
        idxs = [self.word_to_ix.get(w, word_to_ix["<UNK>"]) for w in text.lower().split()]
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Collate Function: 负责将一个 Batch 的数据整理成 Tensor
# 核心作用：Padding (填充)
def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    # pad_sequence 会自动找到 batch 中最长的句子，把短句子用 padding_value (0) 填补
    # batch_first=True -> 输出形状: (Batch_Size, Max_Seq_Len)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=word_to_ix["<PAD>"])
    
    return padded_seqs, torch.stack(labels)

# 3. 定义 LSTM 模型
class TextClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TextClassifierLSTM, self).__init__()
        
        # Embedding Layer (词嵌入层)
        # 作用：将整数索引 (如 5) 转换为一个稠密的实数向量 (如 [0.1, -0.5, ...])
        # padding_idx=0: 告诉模型索引 0 是填充的，不需要计算梯度，向量固定为 0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM Layer
        # batch_first=True: 输入格式为 (Batch, Seq, Feature)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 全连接层: 将 LSTM 的输出映射到分类类别 (2类)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x) 
        # embeds: (batch_size, seq_len, embed_dim)
        
        # LSTM 输出
        # lstm_out: 每个时间步的输出
        # (h_n, c_n): 最后一个时间步的 隐状态 (Hidden State) 和 细胞状态 (Cell State)
        lstm_out, (h_n, c_n) = self.lstm(embeds)
        
        # 我们取最后一个时间步的隐状态 h_n 作为整句话的特征表示
        # h_n shape: (num_layers, batch_size, hidden_dim) -> 我们取最后一层 [-1]
        final_feature = h_n[-1] 
        
        logits = self.fc(final_feature)
        return logits

# 4. 训练逻辑
def train_model():
    EMBED_DIM = 10    # 每个词用 10 维向量表示
    HIDDEN_DIM = 16   # LSTM 记忆单元的大小
    OUTPUT_DIM = 2    # 分类数
    BATCH_SIZE = 4 
    LEARNING_RATE = 0.01
    EPOCHS = 50

    dataset = SentimentDataset(raw_data, word_to_ix)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = TextClassifierLSTM(len(word_to_ix), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training with Batches...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (text_batch, label_batch) in enumerate(dataloader):
            text_batch, label_batch = text_batch.to(device), label_batch.to(device)
            
            model.zero_grad()
            tag_scores = model(text_batch)
            loss = loss_function(tag_scores, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(dataloader):.4f}")
            
    return model

def predict(model, test_sentence):
    model.eval()
    with torch.no_grad():
        idxs = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in test_sentence.lower().split()]
        inputs = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)
        output = model(inputs)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(output, dim=1).item()
        
        label = "Positive" if pred_idx == 1 else "Negative"
        confidence = probs[0][pred_idx].item()
        print(f"Sentence: '{test_sentence}' -> {label} ({confidence:.2f})")

if __name__ == "__main__":
    model = train_model()
    print("\n--- Testing Model ---")
    predict(model, "I love this code")
    predict(model, "This experience is bad")
