import json
import os
import sys

CHECKPOINT_FILE = "benchmark_checkpoint_bk261006.json"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"提示: 找不到断点文件 {CHECKPOINT_FILE}，当前没有任何缓存。")
        return {}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"成功更新断点文件: {CHECKPOINT_FILE}")


def list_status():
    data = load_checkpoint()
    if not data:
        return
    print("\n" + "=" * 60)
    print("               当前断点文件缓存状态概览               ")
    print("=" * 60)
    for task_key, records in data.items():
        print(f"🔹 缓存键名: {task_key}")
        print(f"   └─ 已缓存样本数: {len(records)} 题")
    print("=" * 60)


def clear_target(keyword):
    """
    根据关键字精准清除缓存
    例如输入 'CoT' 会清空所有带 CoT 的策略缓存
    输入 'commonsenseqa' 会清空所有包含该任务的缓存
    """
    data = load_checkpoint()
    if not data:
        return

    new_data = {}
    removed_count = 0
    for task_key, records in data.items():
        if keyword in task_key:
            print(f"[-] 正在清除匹配项: {task_key} ({len(records)} 题)")
            removed_count += 1
        else:
            new_data[task_key] = records

    if removed_count > 0:
        save_checkpoint(new_data)
        print(f"清理完成！共删除了 {removed_count} 个匹配的缓存项。")
    else:
        print(f"未找到包含关键字 '{keyword}' 的缓存项。")


def clear_all():
    if os.path.exists(CHECKPOINT_FILE):
        confirm = input("⚠️ 警告：确定要清空所有断点缓存吗？(y/n): ")
        if confirm.lower() == 'y':
            os.remove(CHECKPOINT_FILE)
            print("已成功删除整个断点文件，所有任务将从头开始！")
    else:
        print("断点文件本身就不存在。")


if __name__ == "__main__":
    while True:
        print("\n--- LLM Benchmark 断点管理器 ---")
        print("1. 查看当前所有缓存状态 (List)")
        print("2. 按关键字精准清除缓存（如清空某个 Prompt 策略或任务）(Clear by keyword)")
        print("3. 清空所有缓存 (Clear All)")
        print("4. 退出 (Exit)")

        choice = input("请选择操作编号 (1-4): ").strip()

        if choice == "1":
            list_status()
        elif choice == "2":
            kw = input("请输入要匹配的关键字（例如策略名如 'CoT' 或任务名如 'piqa'）: ").strip()
            if kw:
                clear_target(kw)
        elif choice == "3":
            clear_all()
        elif choice == "4":
            print("退出管理器。")
            break
        else:
            print("无效输入，请重新选择。")