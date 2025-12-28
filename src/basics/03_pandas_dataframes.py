"""
03_pandas_dataframes.py - Pandas 数据处理实战

【学习目标】
1. 掌握 Pandas 的核心数据结构: DataFrame (表格) 和 Series (单列)。
2. 学会数据的读取、概览、筛选 (loc/iloc) 和清洗 (处理缺失值)。
3. 理解 GroupBy (分组聚合) —— 类似 SQL 的功能。
4. 掌握 Apply 函数进行自定义数据变换。

【Pandas 在 AI 中的角色】
Pandas 是数据预处理 (Data Preprocessing) 的瑞士军刀。
在将数据送入神经网络之前，我们通常需要用 Pandas 进行清洗、特征工程 (Feature Engineering)。
"""

import pandas as pd
import numpy as np

def print_header(msg: str):
    print(f"\n{'='*20} {msg} {'='*20}")

def create_and_inspect_data():
    print_header("1. 数据创建与查看")
    
    # 模拟一份学生成绩表
    # 字典的 Key 是列名，Value 是数据列表
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Age': [20, 21, 22, 20, 21, np.nan], # np.nan 代表缺失值 (Not a Number)
        'Subject': ['Math', 'Physics', 'Math', 'Physics', 'Math', 'Physics'],
        'Score': [85, 92, 78, 88, np.nan, 95],
        'City': ['Beijing', 'Shanghai', 'Beijing', 'Shenzhen', 'Shanghai', 'Beijing']
    }
    df = pd.DataFrame(data)
    
    print("DataFrame (前5行 head):\n", df.head())
    
    print("\n--- 数据基本信息 (info) ---")
    # info() 非常重要，能看到每列的数据类型和非空值数量
    df.info()
    
    print("\n--- 统计摘要 (describe) ---")
    # describe() 自动计算数值列的 均值、标准差、最小/最大值、分位数
    print(df.describe())
    
    return df

def selection_and_filtering(df: pd.DataFrame):
    print_header("2. 数据筛选与索引")
    
    # 1. 选择列
    # df['Name'] 返回的是一个 Series (带索引的一维数组)
    print("选择 'Name' 列:\n", df['Name'].head(3))
    
    # 2. loc (基于标签 Label 的索引) vs iloc (基于位置 Integer Position 的索引)
    # 语法: [行, 列]
    print(f"\nloc[0, 'Name']: {df.loc[0, 'Name']}") # 第0行，Name列
    print(f"iloc[0, 0]    : {df.iloc[0, 0]}")     # 第0行，第0列
    
    # 3. 布尔索引 (核心)
    # 筛选 Score 大于 80 的行
    print("\n筛选 Score > 80 的学生:")
    high_scorers = df[df['Score'] > 80]
    print(high_scorers)
    
    # 多条件筛选: 使用 & (且), | (或)
    print("\n筛选 Subject='Math' 且 Score > 80:")
    math_high = df[(df['Subject'] == 'Math') & (df['Score'] > 80)]
    print(math_high)

def data_cleaning(df: pd.DataFrame):
    print_header("3. 数据清洗 (处理缺失值)")
    
    # 检查每列有多少个缺失值
    print("各列缺失值数量:\n", df.isnull().sum())
    
    # 策略 A: 填充缺失值 (Imputation)
    # 用中位数填补年龄，用均值填补分数
    df_cleaned = df.copy()
    
    median_age = df_cleaned['Age'].median()
    df_cleaned['Age'] = df_cleaned['Age'].fillna(median_age)
    
    mean_score = df_cleaned['Score'].mean()
    df_cleaned['Score'] = df_cleaned['Score'].fillna(mean_score)
    
    print("\n填充缺失值后:\n", df_cleaned)
    
    # 策略 B: 删除含有缺失值的行
    # df.dropna(inplace=True) 
    
    return df_cleaned

def aggregation_and_grouping(df: pd.DataFrame):
    print_header("4. 数据聚合 (GroupBy)")
    
    # 类似 SQL: SELECT Subject, AVG(Score) FROM table GROUP BY Subject
    print("按科目 (Subject) 计算平均分:")
    subject_stats = df.groupby('Subject')['Score'].mean()
    print(subject_stats)
    
    print("\n按城市 (City) 分组，统计 Score 的均值 (mean) 和人数 (count):")
    # agg() 可以同时应用多个聚合函数
    city_stats = df.groupby('City')['Score'].agg(['mean', 'count'])
    print(city_stats)

def advanced_operations(df: pd.DataFrame):
    print_header("5. 进阶操作 (Apply)")
    
    # Apply: 将一个函数应用到 DataFrame 的每一行或每一列
    # 这是一个非常灵活的特征工程工具
    
    def get_grade(score):
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        else: return 'C'
    
    print("使用 apply 生成 'Grade' (等级) 列:")
    df['Grade'] = df['Score'].apply(get_grade)
    print(df[['Name', 'Score', 'Grade']])
    
    # 排序
    print("\n按 Score 降序排列:")
    print(df.sort_values(by='Score', ascending=False))

if __name__ == "__main__":
    print("开始 Pandas 数据处理复习...")
    
    # 1. 加载
    df = create_and_inspect_data()
    
    # 2. 筛选
    selection_and_filtering(df)
    
    # 3. 清洗
    df_cleaned = data_cleaning(df)
    
    # 4. 聚合
    aggregation_and_grouping(df_cleaned)
    
    # 5. 进阶
    advanced_operations(df_cleaned)

    # 6. 保存结果
    from src.utils.config import OUTPUT_DIR
    import os
    output_path = os.path.join(OUTPUT_DIR, "cleaned_student_scores.csv")
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n[Save] 清洗后的数据已保存至: {output_path}")