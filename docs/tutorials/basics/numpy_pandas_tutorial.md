# Python 数据科学基石：Numpy 与 Pandas

## 1. Numpy：高性能矩阵计算

Numpy (Numerical Python) 是 Python 数据科学生态的基石。为什么我们不能只用 Python 的 List？

### 核心区别：List vs Array
*   **存储方式**：
    *   `List`: 存储的是对象的**指针**。这很灵活（可以存整数、字符串混在一起），但读取时不连续，缓存命中率低。
    *   `Array`: 在内存中是一块**连续**的区域，类似 C 语言的数组。CPU 读取非常快。
*   **计算效率**：
    *   `List`: 必须用 `for` 循环逐个处理元素。
    *   `Array`: 支持**向量化 (Vectorization)**，底层指令集并行计算。

### 核心概念：广播 (Broadcasting)
广播是 Numpy 最神奇的特性，它允许不同形状的数组直接相加。

**例子**：
```python
A = [[1, 2, 3],
     [4, 5, 6]]  # Shape (2, 3)

b = [10, 20, 30] # Shape (3,)
```
当计算 `A + b` 时，Numpy 发现 `b` 的维度少，会自动补全为 `(1, 3)`，然后发现行数只有 1，会自动**复制**这一行变成 `(2, 3)`，最后与 A 对应位相加。

这避免了我们手动写循环去复制数据，既简洁又高效。

---

## 2. Pandas：Python 版的 Excel

如果说 Numpy 是为了数学计算，Pandas 就是为了**处理表格数据**。它构建在 Numpy 之上。

### 核心数据结构
1.  **DataFrame**: 二维表格（就像 Excel 的 Sheet）。有行索引 (Index) 和列名 (Columns)。
2.  **Series**: 一列数据。DataFrame 的每一列取出来就是一个 Series。

### 数据分析流水线 (Pipeline)
通常的数据处理步骤如下：

1.  **读取 (I/O)**: `pd.read_csv()`, `pd.read_excel()`
2.  **概览 (Inspection)**:
    *   `df.head()`: 看前几行。
    *   `df.info()`: 看有没有缺失值，数据类型对不对（比如数字读成了字符串）。
    *   `df.describe()`: 看均值、标准差，快速了解数据分布。
3.  **清洗 (Cleaning)**:
    *   `df.dropna()`: 删掉脏数据。
    *   `df.fillna()`: 填补缺失值（通常用平均值或中位数）。
4.  **操作 (Manipulation)**:
    *   `df[df['Score'] > 80]`: 像 SQL 一样筛选数据。
    *   `df.groupby('City')['Score'].mean()`: 透视表操作。

### 学习建议
请直接运行 `src/basics/02_numpy_arrays.py` 和 `src/basics/03_pandas_dataframes.py`。我在代码中添加了大量注释，你可以尝试修改里面的数字，看看输出有什么变化。
