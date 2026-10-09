import time
import tqdm
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import warnings
import datasets
import re
import json


def infer_baseline(sample, model, tokenizer, device):
    question = sample["question"]
    options = sample["options"]

    # 动态判断是否为 DeepSeek 推理模型
    model_name = getattr(model.config, "_name_or_path", "").lower()
    is_deepseek = "r1" in model_name or "reasoner" in model_name

    prompt = f"{question}\n"
    for option in options:
        prompt += f"{option['label']}: {option['text']}\n"

    # 如果是 DeepSeek，追加一句精简思考的提示词引导
    if is_deepseek:
        prompt += "\nPlease keep your thinking process extremely concise and brief.\n"

    prompt += "Answer: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    max_tokens = 1024 if is_deepseek else 5

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2
    )

    input_len = inputs["input_ids"].shape[-1]
    raw_output = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    return raw_output

def infer_pe(sample, model, tokenizer, device):
    question = sample["question"]
    options = sample["options"]
    valid_labels = [str(option["label"]).strip().upper() for option in options]

    # 动态判断是否为 DeepSeek 推理模型
    model_name = getattr(model.config, "_name_or_path", "").lower()
    is_deepseek = "r1" in model_name or "reasoner" in model_name

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

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_message

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_tokens = 1024 if is_deepseek else 5

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2
    )

    input_len = inputs["input_ids"].shape[-1]
    raw_output = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    return raw_output


def infer_fewshot(sample, model, tokenizer, device):
    question = sample["question"]
    options = sample["options"]
    valid_labels = [str(option["label"]).strip().upper() for option in options]

    # 动态判断是否为 DeepSeek 推理模型
    model_name = getattr(model.config, "_name_or_path", "").lower()
    is_deepseek = "r1" in model_name or "reasoner" in model_name

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

    #deepseek_hint = "Please keep your thinking process extremely concise and brief.\n" if is_deepseek else ""

    user_message = (
        #f"{deepseek_hint}"
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

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_message

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_tokens = 1024 if is_deepseek else 5

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2
    )

    input_len = inputs["input_ids"].shape[-1]
    raw_output = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    return raw_output


def get_model_type(model_name: str) -> str:
    """
    根据模型名称自动分类，留给未来测试更多模型的路由接口
    """
    name_lower = model_name.lower()

    # 1. DeepSeek-R1 或 Reasoning 强推理模型类
    if "r1" in name_lower or "reasoner" in name_lower:
        return "reasoning_r1"

    # 2. Qwen/Llama/Mistral 等标准 Chat/Instruct 指令模型类
    elif "instruct" in name_lower or "chat" in name_lower:
        return "instruct"

    # 3. Base 基座/续写模型类 (如 TinyLlama)
    else:
        return "base"


def build_cot_prompt_and_stops(model_name: str, sample: dict, tokenizer):
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
        stop_words = ["<|endoftext|>", "<|im_end|>", "\n\nQuestion:", "\nUser:"]
        prefix_to_append = ""

    elif model_type == "instruct":
        messages = [
            {"role": "user", "content": shot_user},
            {"role": "assistant", "content": shot_assistant},
            {"role": "user", "content": current_user}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stop_words = ["<|endoftext|>", "<|im_end|>", "\nUser:", "\nQuestion:", "\n\n"]
        prefix_to_append = ""

    else:
        one_shot_demo = f"{shot_user}\n{shot_assistant}\n\n"
        prompt = one_shot_demo + current_user + "\nReasoning:"
        stop_words = ["\nHuman:", "\nQuestion:", "Question:", "\n\n"]
        prefix_to_append = "Reasoning: "

    return prompt, stop_words, prefix_to_append


def infer_cot(sample, model, tokenizer, device):
    model_name = getattr(model.config, "_name_or_path", "unknown")
    prompt, stop_words, prefix_to_append = build_cot_prompt_and_stops(model_name, sample, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_tokens = 1024

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        do_sample=False,
        stop_strings=stop_words,
        tokenizer=tokenizer,
    )

    input_len = inputs["input_ids"].shape[-1]
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    raw_output = prefix_to_append + generated_text

    return raw_output

def evaluate_samples_with_checkpoint(
        samples,
        model,
        tokenizer,
        device,
        model_name,
        task_name,
        output_txt_file,
        infer_func,
        strategy_name,
        NUM_SAMPLES_TO_SAVE,
        checkpoint_file="benchmark_checkpoint.json"
):
    """
    带安全原子写入（Atomic Write）的断点续跑函数
    """
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

    for i, sample in enumerate(samples):
        if raw_outputs[i] is not None:
            continue

        raw_output = infer_func(sample, model, tokenizer, device)
        raw_outputs[i] = raw_output

        if task_key not in checkpoint_data:
            checkpoint_data[task_key] = {}

        checkpoint_data[task_key][str(i)] = {
            "question": sample["question"],
            "options": sample["options"],
            "answerKey": sample["answerKey"],
            "raw": raw_output
        }

        # === 核心修改：使用原子写入保护 JSON 文件 ===
        tmp_checkpoint_file = checkpoint_file + ".tmp"
        try:
            with open(tmp_checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # 强制刷入磁盘
            # 原子替换，瞬间生效，绝不会产生截断文件
            os.replace(tmp_checkpoint_file, checkpoint_file)
        except Exception as e:
            print(f"[错误] 写入检查点失败: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 写入日志文件（保持不变）
    with open(output_txt_file, "a", encoding="utf-8") as f:
        f.write("\n==================================================\n")
        f.write(f"模型: {model_name} | 策略: {strategy_name}\n")
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

# 实验问题记录
"""
1. deepseek 自带思考链 放宽了其token限制
2. pe 方法用了更加明确的 prompt
3. few shot 方法 使用明确 prompt+ 与数据集格式一致的题目样例
4. CoT 方法使用step by step 来诱导输出思维链，各个模型都使用参数惩罚了循环输出
5. deepseek 的 reasoning 太 TM 长了！
"""

if __name__ == "__main__":

    FAST_TEST = False # 准备跑全量时，请务必设为 False
    NUM_SAMPLES_TO_SAVE = 2 if FAST_TEST else 20
    fast_sample=2
    checkpoint_file="benchmark_checkpoint_ds.json"
    summary_txt_file = "summary_results.txt"
    ALL_STRATEGIES = {
        "Baseline": infer_baseline,
        "PE": infer_pe,
        "FewShot": infer_fewshot,
        "CoT": infer_cot,
    }
    methods_to_run = ["Baseline", "PE", "FewShot"]
    #methods_to_run = ["CoT"]

    strategy_txt_files = {}
    for strategy in methods_to_run:
        filepath = f"evaluation_samples_{strategy.lower()}.txt"
        strategy_txt_files[strategy] = filepath
        # 注意：如果是断点续跑，不要以 "w" 模式清空日志，否则会把之前跑完的内容覆盖！
        # 这里改用 "a"（追加）模式，或者仅在文件不存在时写入表头
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("==================================================\n")
                f.write(f"      大语言模型 BenchMark 评估记录 ({strategy})      \n")
                f.write("==================================================\n\n")

    #target_tasks = ['commonsenseqa', 'openbookqa', 'piqa']
    target_tasks = ['piqa']

    model_list = [
        #"TinyLlama/TinyLlama_v1.1",
        #"Qwen/Qwen2.5-3B-Instruct",
        #"Qwen/Qwen3-4B-Instruct-2507",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ]

    eval_set = datasets.load_dataset("Open-Style/Open-LLM-Benchmark", "questions")
    grouped_datasets = {}
    for example in eval_set['train']:
        dataset_key = example["dataset"].lower()
        if dataset_key not in grouped_datasets:
            grouped_datasets[dataset_key] = []
        grouped_datasets[dataset_key].append(example)

    #final_summary = {strategy: {m: {} for m in model_list} for strategy in methods_to_run}

    # 模型主循环
    for model_name in model_list:
        print("\n" + "="*50)
        print(f"开始评估模型: {model_name}")
        print("="*50)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16
        )
        device = model.device
        # 消除 warning
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

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

                    # 调用带断点续跑的新函数
                    acc = evaluate_samples_with_checkpoint(
                        test_samples, model, tokenizer, device,
                        model_name, task_name, txt_file, infer_func,
                        strategy_name, NUM_SAMPLES_TO_SAVE,checkpoint_file
                    )
                    #final_summary[strategy_name][model_name][task_name] = acc
            else:
                print(f"警告: 数据集中没有找到 {task_name}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()