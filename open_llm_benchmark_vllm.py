import time
import os
import warnings
import datasets
import re
import json
from vllm import LLM, SamplingParams
# ==================== 1. 各个独立的推理策略函数（保留你的原汁原味） ====================
def infer_baseline(sample, tokenizer, model_name):
    question = sample["question"]
    options = sample["options"]
    is_deepseek = "r1" in model_name.lower() or "reasoner" in model_name.lower()

    prompt = f"{question}\n"
    for option in options:
        prompt += f"{option['label']}: {option['text']}\n"
    prompt += "Answer: "

    max_tokens = 1024 if is_deepseek else 5
    return prompt, max_tokens, ""


def infer_pe(sample, tokenizer, model_name):
    question = sample["question"]
    options = sample["options"]
    valid_labels = [str(option["label"]).strip().upper() for option in options]
    is_deepseek = "r1" in model_name.lower() or "reasoner" in model_name.lower()

    options_text = "\n".join(
        f"{option['label']}: {option['text']}" for option in options
    )
    labels_text = "/".join(valid_labels)
    deepseek_hint = "Please keep your thinking process extremely concise and brief.\n" if is_deepseek else ""

    user_message = (
        f"{deepseek_hint}"
        "You are answering a multiple-choice question.\n"
        "Choose the single best answer based on the question and the options.\n"
        f"Return exactly one of these labels: {labels_text}.\n"
        "Return the label only. Do not explain your answer.\n\n"
        f"Question: {question}\n"
        f"{options_text}\n"
        "Answer:"
    )

    if tokenizer and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_message

    max_tokens = 1024 if is_deepseek else 5
    return prompt, max_tokens, ""


def infer_fewshot(sample, tokenizer, model_name):
    question = sample["question"]
    options = sample["options"]
    valid_labels = [str(option["label"]).strip().upper() for option in options]
    is_deepseek = "r1" in model_name.lower() or "reasoner" in model_name.lower()

    options_text = "\n".join(
        f"{option['label']}: {option['text']}" for option in options
    )
    labels_text = "/".join(valid_labels)

    fewshot_examples = (
        "Question: Which animal is a mammal?\n"
        "A: Eagle\n"
        "B: Dog\n"
        "C: Snake\n"
        "D: Frog\n"
        "Answer: B\n\n"
        "Question: What is the main source of energy for Earth?\n"
        "A: Moon\n"
        "B: Mars\n"
        "C: Sun\n"
        "D: Wind\n"
        "Answer: C\n"
        "Question: What should you use to dry a wet floor?\n"
        "A: A towel\n"
        "B: A book\n"
        "C: A pillow\n"
        "D: A plate\n"
        "Answer: A"
    )

    deepseek_hint = "Please keep your thinking process extremely concise and brief.\n" if is_deepseek else ""

    user_message = (
        f"{deepseek_hint}"
        "You are answering multiple-choice questions.\n"
        "For each question, choose the single best answer based on the question "
        "and the options.\n"
        "Return exactly one option label and nothing else. Do not explain.\n\n"
        f"{fewshot_examples}\n"
        f"For the next question, choose exactly one of these labels: {labels_text}.\n"
        f"Question: {question}\n"
        f"{options_text}\n"
        "Answer:"
    )

    if tokenizer and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_message

    max_tokens = 1024 if is_deepseek else 5
    return prompt, max_tokens, ""


def get_model_type(model_name: str) -> str:
    name_lower = model_name.lower()
    if "r1" in name_lower or "reasoner" in name_lower:
        return "reasoning_r1"
    elif "instruct" in name_lower or "chat" in name_lower:
        return "instruct"
    else:
        return "base"


def infer_cot(sample, tokenizer, model_name):
    model_type = get_model_type(model_name)
    question = sample["question"]
    options_text = "\n".join([f"{opt['label']}: {opt['text']}" for opt in sample["options"]])

    shot_user = (
        "Question: Which of the following is a renewable energy source?\n"
        "A: Coal\nB: Solar power\nC: Natural gas\nD: Petroleum\n"
        "Think step by step in <think> tags briefly to evaluate the options. Be extremely concise. Your analysis must not exceed 3 sentences. You MUST choose EXCLUSIVELY ONE option. Strictly end with 'Final Answer: [LETTER]' (e.g. Final Answer: A)."
    )

    shot_assistant = (
        "<think>\n"
        "1. Coal, natural gas, and petroleum are finite fossil fuels.\n"
        "2. Solar power harnesses energy from the sun, which is inexhaustible.\n"
        "3. Therefore, solar power is the correct renewable source.\n"
        "</think>\n"
        "Final Answer: B"
    )

    current_user = (
        f"Question: {question}\n"
        f"{options_text}\n"
        "Think step by step in <think> tags briefly to evaluate the options. Be extremely concise. Your analysis must not exceed 3 sentences. "
        "You MUST choose EXCLUSIVELY ONE option. Strictly end with 'Final Answer: [LETTER]' (e.g. Final Answer: A)."
    )

    if model_type == "reasoning_r1":
        messages = [
            {
                "role": "user",
                "content": (
                    "Please keep your thinking process extremely concise and brief.\n"
                    f"Question: {question}\n{options_text}\n"
                    "Think step by step in <think> tags briefly to evaluate the options. Be extremely concise. Your analysis must not exceed 3 sentences. "
                    "After reasoning, you MUST choose EXCLUSIVELY ONE option. Strictly end with 'Final Answer: [LETTER]' (e.g. Final Answer: A)."
                )
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_to_append = ""

    elif model_type == "instruct":
        messages = [
            {"role": "user", "content": shot_user},
            {"role": "assistant", "content": shot_assistant},
            {"role": "user", "content": current_user}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_to_append = ""

    else:
        one_shot_demo = f"{shot_user}\n{shot_assistant}\n\n"
        prompt = one_shot_demo + current_user + "\nReasoning:"
        prefix_to_append = "Reasoning: "

    return prompt, 1024, prefix_to_append


# 统一的策略注册表：如果后面要加新方法（比如 NewMethod），只需要写个 infer_newmethod 然后加到这里即可！
ALL_STRATEGIES = {
    "Baseline": infer_baseline,
    "PE": infer_pe,
    "FewShot": infer_fewshot,
    "CoT": infer_cot,
}


# ==================== 2. 评估与断点续跑核心函数 ====================

def evaluate_samples_with_vllm(
        samples,
        llm,
        tokenizer,
        model_name,
        task_name,
        output_txt_file,
        infer_func,
        strategy_name,
        NUM_SAMPLES_TO_SAVE,
        checkpoint_file="benchmark_checkpoint.json"
):
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
        except json.JSONDecodeError:
            print(f"[警告] 检测到旧的 checkpoint 文件已损坏！正在尝试备份并重新开始当前任务...")
            if os.path.exists(checkpoint_file + ".bak"):
                os.remove(checkpoint_file + ".bak")
            os.rename(checkpoint_file, checkpoint_file + ".bak")
            checkpoint_data = {}
    else:
        checkpoint_data = {}

    task_key = f"{model_name}__{strategy_name}__{task_name}"
    completed_records = checkpoint_data.get(task_key, {})

    raw_outputs = [None] * len(samples)
    for idx_str, res in completed_records.items():
        idx = int(idx_str)
        if idx < len(samples):
            raw_outputs[idx] = res["raw"]

    total = len(samples)
    start_time = time.time()

    # 找出所有未完成的样本索引
    pending_indices = [i for i in range(total) if raw_outputs[i] is None]

    if pending_indices:
        prompts_to_generate = []
        prefixes = []
        max_tokens_list = []

        # 统一通过对应的 infer 函数提取 Prompt 和参数
        for idx in pending_indices:
            sample = samples[idx]
            prompt, max_tokens, prefix_to_append = infer_func(sample, tokenizer, model_name)
            prompts_to_generate.append(prompt)
            prefixes.append(prefix_to_append)
            max_tokens_list.append(max_tokens)

        target_max_tokens = max(max_tokens_list) if max_tokens_list else 1024

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=target_max_tokens,
            repetition_penalty=1.2
        )

        print(f"[{strategy_name}] 正在通过 vLLM 批量推理 {len(prompts_to_generate)} 个样本...")

        # 一次性丢进 vLLM 享受极速批量推理
        vllm_outputs = llm.generate(prompts_to_generate, sampling_params)

        if task_key not in checkpoint_data:
            checkpoint_data[task_key] = {}

        for i, sample_idx in enumerate(pending_indices):
            generated_text = vllm_outputs[i].outputs[0].text.strip()
            raw_output = prefixes[i] + generated_text
            raw_outputs[sample_idx] = raw_output

            checkpoint_data[task_key][str(sample_idx)] = {
                "question": samples[sample_idx]["question"],
                "options": samples[sample_idx]["options"],
                "answerKey": samples[sample_idx]["answerKey"],
                "raw": raw_output
            }

        # 原子写入 JSON
        tmp_checkpoint_file = checkpoint_file + ".tmp"
        try:
            with open(tmp_checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_checkpoint_file, checkpoint_file)
        except Exception as e:
            print(f"[错误] 写入检查点失败: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 写入日志文件
    with open(output_txt_file, "a", encoding="utf-8") as f:
        f.write("\n==================================================\n")
        f.write(f"模型: {model_name} | 策略: {strategy_name} (vLLM)\n")
        f.write(f"任务: {task_name}\n")
        f.write(f"生成完毕样本数: {total} | 用时: {elapsed_time:.2f}s\n")
        f.write("--------------------------------------------------\n")

        for i in range(min(NUM_SAMPLES_TO_SAVE, total)):
            sample = samples[i]
            raw = raw_outputs[i]
            target = sample["answerKey"]
            options_text = " | ".join(
                [f"{opt['label']}: {opt['text']}" for opt in sample["options"]]
            )

            f.write(f"[样例 {i + 1}]\n")
            f.write(f"题目: {sample['question']}\n")
            f.write(f"选项: {options_text}\n")
            f.write(f"原始生成: '{raw}'\n")
            f.write(f"标准答案: '{target}'\n")
            f.write("-" * 40 + "\n")

    print(f"[{strategy_name}] 任务 {task_name} 原始文本生成完成 | 耗时: {elapsed_time:.2f}s")


# ==================== 3. 主程序循环 ====================

if __name__ == "__main__":
    FAST_TEST = False  # 跑全量时设为 False
    NUM_SAMPLES_TO_SAVE = 2 if FAST_TEST else 20
    fast_sample = 2

    # 在这里控制你想跑哪些方法
    methods_to_run = ["Baseline", "PE", "FewShot"]
    # methods_to_run = ["CoT"]

    strategy_txt_files = {}
    for strategy in methods_to_run:
        filepath = f"evaluation_samples_{strategy.lower()}_test.txt"
        strategy_txt_files[strategy] = filepath
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("==================================================\n")
                f.write(f"      大语言模型 BenchMark 评估记录 ({strategy})      \n")
                f.write("==================================================\n\n")

    target_tasks = ['commonsenseqa', 'openbookqa', 'piqa']

    model_list = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ]

    eval_set = datasets.load_dataset("Open-Style/Open-LLM-Benchmark", "questions")
    grouped_datasets = {}
    for example in eval_set['train']:
        dataset_key = example["dataset"].lower()
        if dataset_key not in grouped_datasets:
            grouped_datasets[dataset_key] = []
        grouped_datasets[dataset_key].append(example)

    # 模型主循环
    for model_name in model_list:
        print("\n" + "=" * 50)
        print(f"开始使用 vLLM 评估模型: {model_name}")
        print("=" * 50)

        # 初始化 vLLM 引擎
        llm = LLM(
            model=model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90
        )
        tokenizer = llm.get_tokenizer()

        for task_name in target_tasks:
            if task_name in grouped_datasets:
                print(f"\n---> 任务: {task_name}")
                test_samples = grouped_datasets[task_name][:fast_sample] if FAST_TEST else grouped_datasets[task_name]
                all_len = len(grouped_datasets[task_name])
                eva_len = len(test_samples)
                print("全部数据集长度:", all_len)
                print("实际使用数据集长度:", eva_len)

                for strategy_name in methods_to_run:
                    infer_func = ALL_STRATEGIES[strategy_name]
                    txt_file = strategy_txt_files[strategy_name]

                    # 调用评估函数
                    evaluate_samples_with_vllm(
                        test_samples, llm, tokenizer,
                        model_name, task_name, txt_file,
                        infer_func, strategy_name, NUM_SAMPLES_TO_SAVE
                    )
            else:
                print(f"警告: 数据集中没有找到 {task_name}")

        del llm
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()