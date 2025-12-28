# 自然语言处理：从 RNN 到 Transformer

## 1. 序列问题 (Sequence Modeling)
与图像处理不同，文本处理的核心是**顺序**。
"I ate an apple" 和 "Apple an ate I" 的含义完全不同。

## 2. LSTM (Long Short-Term Memory)
对应代码：`src/nlp/01_lstm_sentiment.py`

### 核心思想
RNN (循环神经网络) 通过一个“隐状态” (Hidden State) 记录历史信息，并传递给下一个时间步。
但普通 RNN 有“遗忘”问题（梯度消失），记不住长句子开头的词。

LSTM 引入了三个“门”：
1.  **遗忘门 (Forget Gate)**: 决定丢弃多少旧信息。
2.  **输入门 (Input Gate)**: 决定吸收多少新信息。
3.  **输出门 (Output Gate)**: 决定输出什么信息。

这让 LSTM 能够捕捉长距离依赖（比如文章开头的“他”，指代几百字后出现的“小明”）。

---

## 3. Transformer：Attention Is All You Need
对应代码：`src/nlp/02_transformer_mechanism.py` 和 `03_transformer_translator.py`

### 为什么取代了 LSTM？
LSTM 必须**串行**计算（必须等 $t-1$ 算完才能算 $t$），导致训练很慢，无法并行。
Transformer 抛弃了循环，提出 **Self-Attention (自注意力)** 机制。

### 核心公式：Query, Key, Value
想象你在图书馆找书：
*   **Query (Q)**: 你手里的借书条（你的需求）。
*   **Key (K)**: 图书馆每本书脊上的标签。
*   **Value (V)**: 书里面的实际内容。

Attention 计算过程：
1.  拿你的 $Q$ 去和所有书的 $K$ 进行比对（点积），算出**相关性分数**。
2.  用这些分数（归一化后）对 $V$ 进行加权求和。
3.  如果分数高，说明这本书是你想要的，它的内容 $V$ 就被更多地提取出来。

### 训练流程 (Seq2Seq)
我们在 `03_transformer_translator.py` 中实现了一个字符串翻转器。
*   **Encoder**: 读入 "a b c"，生成一系列特征向量。
*   **Decoder**:
    *   Step 1: 输入 `<SOS>`, 结合 Encoder 的特征，预测 "c"。
    *   Step 2: 输入 `<SOS> c`, 预测 "b"。
    *   Step 3: 输入 `<SOS> c b`, 预测 "a"。
    *   Step 4: 输入 `<SOS> c b a`, 预测 `<EOS>`。

通过 **Teacher Forcing**，训练时我们直接告诉 Decoder 正确的历史输入，这让模型收敛极快。
