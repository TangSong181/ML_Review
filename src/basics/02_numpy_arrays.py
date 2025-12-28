"""
02_numpy_arrays.py - Numpy 数组操作详解

【学习目标】
1. 理解 Numpy 数组 (ndarray) 与 Python 列表的区别。
2. 掌握数组的创建、形状变换 (Reshape)。
3. 理解广播机制 (Broadcasting) —— Numpy 最强大的特性之一。
4. 熟悉矩阵运算 (点积) 与 聚合操作 (求和、平均值)。

【为什么需要 Numpy?】
Python 原生的列表 (List) 保存的是对象的指针，运算速度慢且内存占用大。
Numpy 底层使用 C 语言编写，数组在内存中是连续存储的，因此计算速度极快，
是所有深度学习框架 (PyTorch, TensorFlow) 的数据基础。
"""

import numpy as np

def print_header(msg: str):
    """辅助函数：打印美观的标题"""
    print(f"\n{'='*20} {msg} {'='*20}")

def basic_operations():
    print_header("1. 基础数组操作")
    
    # 1. 创建数组
    # np.array() 将 Python 列表转换为 Numpy 数组
    # dtype (Data Type) 会自动推断，也可以手动指定 (如 dtype=np.float32)
    data_list = [[1, 2, 3], [4, 5, 6]]
    a: np.ndarray = np.array(data_list)
    
    print(f"数组 A (从列表创建):\n{a}")
    print(f"维度 (Ndim): {a.ndim}")      # 2 (几维数组)
    print(f"形状 (Shape): {a.shape}")     # (2, 3) -> 2行3列
    print(f"元素总数 (Size): {a.size}")   # 6
    print(f"数据类型 (Dtype): {a.dtype}") # usually int32 or int64

    # 2. Reshape (形状变换)
    # 在深度学习中，我们经常需要改变张量的形状。
    # 例如：把 (2, 3) 的图片特征展平成 (6,) 的向量送入全连接层。
    # 注意：Reshape 前后元素总数 (Size) 必须保持一致 (2*3 = 6)。
    b: np.ndarray = a.reshape(3, 2)
    print(f"\nReshape 后 (3行2列):\n{b}")
    
    # -1 代表自动推断该维度大小。例如 reshape(1, -1) 会把数组变成 1 行，列数自动算。
    c: np.ndarray = a.reshape(1, -1)
    print(f"Reshape(1, -1) (展平为行向量):\n{c}")

def broadcasting_review():
    print_header("2. 广播机制 (Broadcasting)")
    """
    广播机制是 Numpy 允许不同形状的数组进行算术运算的规则。
    规则简单来说：如果两个数组的后缘维度 (trailing dimensions) 的轴长度相符，
    或其中一方的长度为 1，则认为它们是兼容的。广播会在缺失或长度为 1 的维度上进行“复制”。
    """
    
    A: np.ndarray = np.array([[1, 2, 3], 
                              [4, 5, 6]]) # Shape: (2, 3)
    
    # 标量广播
    print("A + 10 (标量广播到所有元素):\n", A + 10)
    
    # 数组广播
    # b 的 Shape 是 (3,)
    # 1. 补全维度: (3,) -> (1, 3)
    # 2. 复制维度: (1, 3) -> (2, 3) (沿着轴 0 复制)
    b: np.ndarray = np.array([10, 20, 30]) 
    
    result: np.ndarray = A + b
    print(f"\nA (2,3) + b (3,) (b被广播为2行):\n{result}")

def matrix_multiplication():
    print_header("3. 矩阵运算 vs 逐元素运算")
    
    X: np.ndarray = np.array([[1, 2], [3, 4]])
    Y: np.ndarray = np.array([[5, 6], [7, 8]])
    
    # 1. 逐元素乘法 (Element-wise)
    # 对应位置直接相乘：1*5, 2*6...
    # 在深度学习中，Mask (掩码) 操作常用此方法。
    element_wise = X * Y
    print(f"逐元素乘法 (X * Y):\n{element_wise}")
    
    # 2. 矩阵乘法 (Dot Product / Matrix Multiplication)
    # 线性代数的核心：行 x 列。
    # (m, n) @ (n, p) -> (m, p)
    # 全连接层 (Linear Layer) 的本质就是矩阵乘法。
    dot_prod = np.dot(X, Y) 
    # 或者使用 Python 3.5+ 的 @ 运算符
    dot_prod_at = X @ Y
    
    print(f"\n矩阵乘法 (X @ Y) [线性代数核心]:\n{dot_prod_at}")

def advanced_indexing():
    print_header("4. 高级索引与切片")
    # 创建一个 5x5 的随机整数数组
    data = np.random.randint(0, 100, (5, 5))
    print(f"原始随机数据 (5x5):\n{data}")
    
    # 1. 布尔索引 (Boolean Masking)
    # 生成一个与 data 形状相同的 True/False 矩阵，True 的位置会被选中。
    # 用途：筛选数据、去除异常值。
    mask = data > 50
    print(f"\n大于 50 的元素 (布尔索引):\n{data[mask]}")
    
    # 2. 切片 (Slicing)
    # [start:end, start:end]
    # 取中间的 3x3 区域 (行 1-3, 列 1-3)
    # 注意：Numpy 切片是视图 (View)，修改切片会影响原数组！
    center_patch = data[1:4, 1:4]
    print(f"\n中间 3x3 切片:\n{center_patch}")

def aggregation_and_axis():
    print_header("5. 聚合函数与 Axis (轴)")
    """
    理解 Axis (轴) 是关键：
    - axis=0: 沿着“行”的方向操作 (跨行)，即“压缩”行，保留列。 -> 操作后行数消失。
    - axis=1: 沿着“列”的方向操作 (跨列)，即“压缩”列，保留行。 -> 操作后列数消失。
    """
    
    data = np.array([[1, 2, 3], 
                     [4, 5, 6], 
                     [7, 8, 9]])
    
    print(f"数据:\n{data}")
    
    # 求和
    print(f"所有元素和: {np.sum(data)}") # 45
    print(f"按列求和 (axis=0): {np.sum(data, axis=0)}") # [12 15 18] -> 变成了 1 行 3 列 (行被压扁了)
    print(f"按行求和 (axis=1): {np.sum(data, axis=1)}") # [ 6 15 24] -> 变成了 3 行 1 列 (列被压扁了)

    # Argmax: 返回最大值的索引
    # 在分类任务中，神经网络输出概率向量 [0.1, 0.8, 0.1]，argmax 得到 1 (即类别索引)
    print(f"\n每列最大值的索引 (axis=0): {np.argmax(data, axis=0)}")
    print(f"每行最大值的索引 (axis=1): {np.argmax(data, axis=1)}")

if __name__ == "__main__":
    print("开始 Numpy 深度复习...")
    np.random.seed(42) # 设置随机种子，保证每次运行结果一致
    basic_operations()
    broadcasting_review()
    matrix_multiplication()
    advanced_indexing()
    aggregation_and_axis()