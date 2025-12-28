# 00_python_basics_enhanced.py - Python 基础复习 (Enhanced)

def print_header(msg: str):
    print(f"\n{'='*20} {msg} {'='*20}")

# 1. 变量与数据类型
def variables_and_data_types():
    print_header("1. 变量与数据类型")
    
    # 基本数据类型
    integer_var: int = 10
    float_var: float = 3.14
    string_var: str = "Hello, Python!"
    boolean_var: bool = True
    print(f"整数: {integer_var}, 类型: {type(integer_var)}")
    print(f"浮点数: {float_var}, 类型: {type(float_var)}")
    print(f"字符串: {string_var}, 类型: {type(string_var)}")
    print(f"布尔值: {boolean_var}, 类型: {type(boolean_var)}")

    # 列表 (List) - 有序、可变
    my_list: list[int | str | float | bool] = [1, 2, "three", 4.0, True]
    print(f"列表: {my_list}, 长度: {len(my_list)}")
    my_list.append(5)
    print(f"追加后列表: {my_list}")
    print(f"列表索引 0: {my_list[0]}")

    # 元组 (Tuple) - 有序、不可变
    my_tuple = (10, 20, "thirty")
    print(f"元组: {my_tuple}")

    # 字典 (Dictionary) - 键值对、无序 (Python 3.7+ 保持插入顺序)
    my_dict: dict[str, str | int] = {"name": "Alice", "age": 30, "city": "New York"}
    print(f"字典: {my_dict}")
    print(f"字典键 'name' 的值: {my_dict['name']}")
    my_dict['age'] = 31
    print(f"更新后的字典: {my_dict}")

    # 集合 (Set) - 无序、不重复元素
    my_set = {1, 2, 3, 2, 1}
    print(f"集合: {my_set}")
    my_set.add(4)
    print(f"追加元素后的集合: {my_set}")
    
# 2. 控制流
def control_flow():
    print_header("2. 控制流")

    # if-elif-else
    x = 15
    if x > 20:
        print("x 大于 20")
    elif x > 10:
        print("x 大于 10 但不大于 20")
    else:
        print("x 小于等于 10")

    # for 循环
    print("for 循环遍历列表:")
    for item in ["apple", "banana", "cherry"]:
        print(item)

    print("for 循环遍历数字范围 (0-4):")
    for i in range(5):
        print(i)
        
    print("for 循环遍历字典的键值对:")
    for key, value in {"a": 1, "b": 2}.items():
        print(f"键: {key}, 值: {value}")

    # while 循环
    print("while 循环:")
    count = 0
    while count < 3:
        print(f"计数: {count}")
        count += 1
        
    # 列表推导式 (List Comprehensions) - 高效且简洁
    print("列表推导式:")
    squares: list[int] = [i * i for i in range(5)]
    print(f"0-4 的平方: {squares}")
    
    even_numbers: list[int] = [i for i in range(10) if i % 2 == 0]
    print(f"0-9 的偶数: {even_numbers}")

# 3. 函数 (带类型注解)
def functions_review():
    print_header("3. 函数 (Type Hints)")

    # 定义函数，增加类型注解
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    print(greet("Alice"))

    # 带默认参数的函数
    def power(base: int | float, exp: int = 2) -> int | float:
        return base ** exp

    print(f"5 的平方: {power(5)}")
    print(f"2 的 3 次方: {power(2, 3)}")

    # 可变参数 *args 和 **kwargs
    def log_arguments(*args: int | str, **kwargs: int | str):
        print(f"位置参数 (args): {args}")
        print(f"关键字参数 (kwargs): {kwargs}")

    log_arguments(1, 2, "test", a=10, b="hi")

# 4. 类与对象
class Dog:
    # 构造函数
    def __init__(self, name: str, breed: str):
        self.name: str = name
        self.breed: str = breed
        self.tricks: list[str] = [] # 实例变量

    # 实例方法
    def bark(self) -> str:
        return f"{self.name} 汪汪叫!"

    def add_trick(self, trick: str) -> None:
        self.tricks.append(trick)

def classes_review():
    print_header("4. 类与对象")

    my_dog = Dog("Buddy", "Golden Retriever")
    print(f"我的狗叫 {my_dog.name}, 品种是 {my_dog.breed}")
    print(my_dog.bark())
    my_dog.add_trick("roll over")
    print(f"{my_dog.name} 会的把戏: {my_dog.tricks}")

# --- 新增部分 ---

# 5. 异常处理 (Exception Handling)
def exception_handling():
    print_header("5. 异常处理 (Try-Except)")
    
    inputs: list[str] = ["10", "5", "zero", "2", "0", "abc"]
    
    print(f"尝试将列表元素转换为倒数: {inputs}")
    for item in inputs:
        try:
            val: int = int(item)
            result: float = 100 / val
            print(f"100 / {val} = {result:.2f}")
        except ValueError:
            print(f"错误: '{item}' 不是一个有效的数字，无法转换。")
        except ZeroDivisionError:
            print(f"错误: 不能除以零。")
        except Exception as e:
            # 捕获所有其他可能的异常
            print(f"发生了未知的错误: {e}, 针对输入 '{item}'")
        finally:
            # 这里的代码无论是否出错都会执行
            print(f"--- 处理完输入: '{item}' ---")
    print("异常处理示例结束。")

# 6. 文件操作 (Context Manager)
def file_operations():
    print_header("6. 文件操作 (With Context)")
    
    # Ensure directory exists
    import os
    os.makedirs("ML_Review/outputs/logs", exist_ok=True)
    filename: str = "ML_Review/outputs/logs/test_note.txt"
    
    # 写入文件
    print(f"正在写入文件: {filename} ...")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("这是第一行。\n")
            f.write("Deep Learning Review.\n")
            f.write("这是文件操作的测试内容。\n")
        print(f"文件 '{filename}' 写入成功。")
    except IOError as e:
        print(f"文件写入错误: {e}")
    
    # 读取文件
    print(f"正在读取文件: {filename} ...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content: str = f.read()
            print(f"--- 文件 '{filename}' 内容 ---\n{content}----------------")
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到。")
    except IOError as e:
        print(f"文件读取错误: {e}")
    finally:
        # 清理操作，例如删除临时文件
        # import os
        # if os.path.exists(filename):
        #     os.remove(filename)
        #     print(f"临时文件 '{filename}' 已删除。")
        pass # 这里选择不删除，方便用户查看

if __name__ == "__main__":
    print("开始 Python 基础复习 (Enhanced)...")
    variables_and_data_types()
    control_flow()
    functions_review()
    classes_review()
    exception_handling() # 新增
    file_operations()    # 新增
    print("Python 基础复习 (Enhanced) 结束。")

