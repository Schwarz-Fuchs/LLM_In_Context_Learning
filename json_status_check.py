import json
import os


def check_checkpoint_health(file_path):
    print(f"正在检查文件: {file_path}")

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print("[错误] 文件不存在！")
        return False

    # 2. 检查文件是否为空
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} 字节")
    if file_size == 0:
        print("[错误] 文件为空，可能在创建时被中断！")
        return False

    # 3. 尝试解析 JSON 结构
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("[健康] JSON 语法结构完全正确，可以正常加载！")

        # 4. 可选：针对断点续传的业务逻辑结构检查
        if isinstance(data, dict):
            print(f"-> 顶层模型/任务检查点数量: {len(data)}")
            for model_name, checkpoints in data.items():
                print(f"   - 任务 [{model_name}]: 已记录 {len(checkpoints)} 条进度")
        else:
            print("[警告] 顶层结构不是一个字典 (Dict)，请注意检查。")

        return True

    except json.JSONDecodeError as e:
        print(f"[损坏] JSON 结构受损！这通常是由程序在写入时意外中断导致的。")
        print(f"   - 错误详情: {e.msg}")
        print(f"   - 错在行号: {e.lineno}, 列号: {e.colno}")
        return False
    except Exception as e:
        print(f"[错误] 读取文件时发生未知错误: {e}")
        return False


if __name__ == "__main__":
    # 将这里的路径替换为你的 json 文件名
    check_checkpoint_health("benchmark_checkpoint.json")