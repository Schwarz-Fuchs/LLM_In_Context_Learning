import json
import os
import pandas as pd
import re


def extract_answer(decoded_text: str) -> str:
  if not decoded_text:
    return ""

  text = decoded_text.strip()

  # 优先检查全文最开头（防止模型把答案写在最前面）
  raw_head = text[:10].strip()
  pure_head_match = re.match(r"^([A-E])(?:\.|\b)", raw_head, re.IGNORECASE)
  if pure_head_match:
    return pure_head_match.group(1).upper()

  # 1. 防御：针对 DeepSeek-R1 等 Reasoning 模型，拦截思考未完成的截断样本
  if "<think>" in text and "</think>" not in text:
    return ""
  if "</think>" in text:
    text = text.split("</think>")[-1].strip()

  # 2. 拦截多轮对话续写/自我伪造
  for stop_word in [
      "\nHuman:",
      "\nAssistant:",
      "\nQuestion:",
      "Given the following",
  ]:
    if stop_word in text:
      text = text.split(stop_word)[0].strip()

  # 3. 匹配 LaTeX 格式的 Boxed 答案：如 \boxed{A} 或 $\boxed{A}$
  boxed_match = re.search(
      r"\\boxed\{\s*(?:\\text\{)?([A-E])(?:\})?\}", text, re.IGNORECASE
  )
  if boxed_match:
    return boxed_match.group(1).upper()

  # 【新增规则】专门匹配形如 "The correct answer is **C**." 或 "The best choice is **B**" 的独立加粗字母模式
  isolated_bold_match = re.search(
      r"(?:correct\s+answer\s+is|answer\s+is|choice\s+is|best\s+choice\s+is|best\s+answer\s+is)\b\s*\*\*([A-E])\*\*",
      text,
      re.IGNORECASE,
  )
  if isolated_bold_match:
    return isolated_bold_match.group(1).upper()

  # 4. 匹配长句带描述模式：如 "The correct answer is **C: none of these**" (兼容双星号)
  inline_desc_match = re.search(
      r"(?:correct\s+answer\s+is|answer\s+is|choice\s+is|best\s+choice\s+is|best\s+answer\s+is)\b\s*\*{0,2}\s*([A-E])\s*[:\)]",
      text,
      re.IGNORECASE,
  )
  if inline_desc_match:
    return inline_desc_match.group(1).upper()

  # 5. 匹配标准结构化前缀：如 "Answer: A", "Option: (C)", 兼容各种 Markdown 星号
  match = re.search(
      r"\*{0,2}(?:Answer|Option|Choice|Final Answer)\s*(?:is|:|\=)?\s*\*?\(?([A-E])\)?\*{0,2}",
      text,
      re.IGNORECASE,
  )
  if match:
    return match.group(1).upper()

  # 6. 匹配被加粗且带冒号的独立选项：如 "**C:**" 或 "**c -**"
  bold_colon_match = re.search(
      r"\*\*([A-E])\s*[:\-]\s*\*\*", text, re.IGNORECASE
  )
  if bold_colon_match:
    return bold_colon_match.group(1).upper()

  # 7. 匹配开头紧跟的选项 (如 "A.", "A)", "(A)")
  head_match = re.match(r"^(?:[A-E])(?:\.|\)|\b)", text, re.IGNORECASE)
  if head_match:
    return text[0].upper()

  # 8. 直接回答/开门见山：检查前 30 个字符
  head_text = text[:30].strip()
  head_match = re.search(
      r"^(?:[0-9]+[\.\s]*)?\*?\(?([A-E])\)?\*?[\.\s\:\,]*",
      head_text,
      re.IGNORECASE,
  )
  if head_match:
    return head_match.group(1).upper()

  # 9. 尾部结论词：扩大检索范围到最后 100 个字符
  tail_text = text[-100:]
  tail_match = re.search(
      r"(?:is|choose|select|answer)\s*\*?\(?([A-E])\)?\b",
      tail_text,
      re.IGNORECASE,
  )
  if tail_match:
    return tail_match.group(1).upper()

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


def analyze_mismatches(
    checkpoint_file="benchmark_checkpoint_ds.json",
    output_txt="analysis_result.txt",
):
  """检查 Checkpoint 文件，统计准确率并将错误样例及汇总写入到 TXT 文件中。

  针对长思维链模型，提取失败时优先展示文本尾部结论。
"""
  if not os.path.exists(checkpoint_file):
    print(f"未找到检查点文件: {checkpoint_file}")
    return

  with open(checkpoint_file, "r", encoding="utf-8") as f:
    data = json.load(f)

  with open(output_txt, "w", encoding="utf-8") as out_f:
    for task_key, records in data.items():
      header = f"\n==================== 分析任务: {task_key} ====================\n"
      #print(header.strip())
      out_f.write(header)

      correct = 0
      total = len(records)
      extraction_fail = 0
      model_wrong = 0

      for idx_str, item in records.items():
        answer_key = item.get("answerKey", "").strip().upper()
        raw_text = item.get("raw", "")

        pred = extract_answer(raw_text)

        if not pred:
          extraction_fail += 1
          # 【修改点】由于大模型答案在结尾，这里改为打印【尾部 400 个字符】和【头部 150 个字符】
          head_snippet = raw_text[:150]
          tail_snippet = raw_text[-400:] if len(raw_text) > 400 else raw_text

          msg = (
              f"[提取失败] 样本ID: {idx_str}\n"
              f"   问题: {item.get('question')}\n"
              f"   标准答案: {answer_key}\n"
              f"   Raw 文本【头部】: {repr(head_snippet)}\n"
              f"   Raw 文本【尾部】: {repr(tail_snippet)}\n"
              + "-" * 40
              + "\n"
          )
          #print(msg.strip())
          out_f.write(msg)

        elif pred == answer_key:
          correct += 1
        else:
          model_wrong += 1
          tail_snippet = raw_text[-300:] if len(raw_text) > 300 else raw_text
          msg = (
              f"[模型答错] 样本ID: {idx_str} | 标准答案: {answer_key} |"
              f" 模型预测: {pred}\n"
              f"   问题: {item.get('question')}\n"
              f"   Raw 文本【尾部结论】: {repr(tail_snippet)}\n"
              + "-" * 40
              + "\n"
          )
          #print(msg.strip())
          out_f.write(msg)

      acc_rate = (correct / total * 100) if total > 0 else 0.0
      summary = (
          f"\n--- 统计汇总 ---\n总题数: {total}\n正确数: {correct}"
          f" (准确率: {acc_rate:.2f}%)\n正则提取失败数 (返回空):"
          f" {extraction_fail}\n模型推理错误数: {model_wrong}\n"
      )
      #print(summary.strip())
      out_f.write(summary)

  print(f"\n[提示] 分析报告已成功保存至: {output_txt}")


if __name__ == "__main__":
    #checkpoint_file= "benchmark_checkpoint.json"
    checkpoint_file = ("benchmark_checkpoint.json")
    calculate_summary_from_checkpoint(checkpoint_file)
    analyze_mismatches("benchmark_checkpoint.json")