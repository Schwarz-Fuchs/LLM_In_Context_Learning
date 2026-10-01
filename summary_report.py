import json
import os
import pandas as pd
import re


def extract_answer(decoded_text: str) -> str:
    if not decoded_text:
        return ""

    text = decoded_text.strip()

    # 1. 防御：针对 DeepSeek-R1 等 Reasoning 模型，拦截思考未完成的截断样本
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # 2. 拦截多轮对话续写/自我伪造
    for stop_word in ["\nHuman:", "\nAssistant:", "\nQuestion:", "Given the following"]:
        if stop_word in text:
            text = text.split(stop_word)[0].strip()

    # 【新增优化 1】直接匹配被 Markdown 粗体包裹或带点号的开头：如 "**A.**", "A.", "**A**"
    # 清理掉开头的 Markdown 符号
    clean_text = re.sub(r'[\*\_\`]', '', text).strip()

    # 匹配开头紧跟的选项 (如 "A.", "A)", "(A)", 甚至独立的字母)
    head_match = re.match(r'^(?:[A-E])(?:\.|\)|\b)', clean_text, re.IGNORECASE)
    if head_match:
        # 提取第一个字母
        return clean_text[0].upper()

    # 【优先级 2】标准结构化前缀：如 "Answer: A", "Option: (C)", "Choice = B"
    match = re.search(r"(?:Answer|Option|Choice)\s*(?:is|:|\=)?\s*\*?\(?([A-E])\)?\*?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 【优先级 3】直接回答/开门见山：检查前 20 个字符（放宽对标点和空格的限制）
    head_text = text[:30].strip()
    head_match = re.search(r"^(?:[0-9]+[\.\s]*)?\*?\(?([A-E])\)?\*?[\.\s\:\,]*", head_text, re.IGNORECASE)
    if head_match:
        return head_match.group(1).upper()

    # 【优先级 4】尾部结论词
    tail_text = text[-60:]
    tail_match = re.search(r"(?:is|choose|select)\s*\*?\(?([A-E])\)?\*?", tail_text, re.IGNORECASE)
    if tail_match:
        return tail_match.group(1).upper()

    return ""


def extract_answer_cot(raw_output: str) -> str:
    if not raw_output:
        return ""

    text = raw_output.strip()

    # 1. 拦截 DeepSeek-R1 思考截断
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # 2. 拦截多轮对话/自我续写
    for stop_word in ["\nHuman:", "\nAssistant:", "\nQuestion:", "Given the following"]:
        if stop_word in text:
            text = text.split(stop_word)[0].strip()

    # 3. 严格结构化锚定（支持被 ** 粗体包裹的选项，如 **A** 或 **A.**）
    strict_patterns = [
        r"Final\s*(?:Answer|Choice|Option)\s*[:=]?\s*\*?\(?([A-E])\)?\*?",
        r"(?:The\s*)?(?:correct|final)\s*(?:answer|option|choice)\s*(?:is|:|=)\s*\*?\(?([A-E])\)?\*?",
        r"Answer\s*[:=]\s*\*?\(?([A-E])\)?\*?",
        # 兼容单独成行或带有 Markdown 粗体的格式，如 **A.** 或 **C**
        r"\n\s*\*?\s*([A-E])(?:\.|\))\s*\**",
    ]

    for pattern in strict_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # 4. 尾部推导词锚定（如 "so the answer is c" 或结尾直接给字母）
    tail_text = text[-100:]
    tail_match = re.search(
        r"(?:so|therefore|thus|hence|choose|select)\s*(?:option|choice)?\s*\*?\(?([A-E])\)?\*?",
        tail_text,
        re.IGNORECASE,
    )
    if tail_match:
        return tail_match.group(1).upper()

    # 5. 保底：如果整个文本的最后有效字符是一个孤立的字母（如回答只有 "c" 或 "**c**"）
    cleaned_tail = re.sub(r'[\*\_\`\.\s]', '', text[-10:])
    if cleaned_tail and cleaned_tail[-1].upper() in ['A', 'B', 'C', 'D', 'E']:
        return cleaned_tail[-1].upper()

    return ""


def calculate_summary_from_checkpoint(checkpoint_file="benchmark_checkpoint.json"):
    if not os.path.exists(checkpoint_file):
        print(f"错误: 找不到断点文件 {checkpoint_file}，请确认文件名或路径是否正确。")
        return

    # 1. 加载断点数据
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)

    print("\n" + "=" * 60)
    print("       从断点 JSON 文件中离线统计的准确率汇总报告       ")
    print("=" * 60)

    table_rows = []

    # 2. 遍历 checkpoint 中的每个任务组合 (格式: model_name__strategy_name__task_name)
    for task_key, records in checkpoint_data.items():
        parts = task_key.split("__")
        if len(parts) != 3:
            continue
        model_name, strategy_name, task_name = parts

        total = len(records)
        completed_count = 0
        correct = 0

        # 3. 遍历该任务下的每一个样本
        for idx_str, res in records.items():
            raw_output = res.get("raw", "")
            target = res.get("answerKey", "")

            if raw_output is not None:
                completed_count += 1

                # 根据策略类型选择对应的解析函数
                if "cot" in strategy_name.lower():
                    pred = extract_answer_cot(raw_output)
                else:
                    pred = extract_answer(raw_output)

                # 比对预测结果与标准答案
                if pred and target and pred == target:
                    correct += 1

        accuracy = (correct / completed_count) * 100 if completed_count > 0 else 0

        # 收集表格行数据
        table_rows.append({
            "Model": model_name.split("/")[-1],  # 简写模型名，保持表格清爽
            "Strategy": strategy_name,
            "Task": task_name,
            "Accuracy (%)": f"{accuracy:.2f}%",
            "Correct/Completed": f"{correct}/{completed_count}",
            "Progress": f"{completed_count}/{total}"
        })

    if not table_rows:
        print("暂无有效的评测记录。")
        return

    # 4. 转换为 Pandas DataFrame 并打印美观表格
    df = pd.DataFrame(table_rows)
    df = df[["Model", "Strategy", "Task", "Accuracy (%)", "Correct/Completed", "Progress"]]

    print("\n📊 **综合评测结果表格汇总**：\n")
    print(df.to_string(index=False))
    print("\n" + "=" * 60)
    print("统计完成！")


if __name__ == "__main__":
    checkpoint_file="benchmark_checkpoint.json"
    calculate_summary_from_checkpoint(checkpoint_file)