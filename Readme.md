# Open-LLM Benchmark & In-Context Learning Evaluation

本项目旨在评估多款中小型开源大语言模型在不同提示词策略（0-shot、Few-shot、简单 CoT）下的上下文学习（In-Context Learning, ICL）表现与推理能力。项目集成了多任务支持、原子写入的断点续跑机制以及自动化的结果解析汇总工具。

---

## 项目结构

* **`open_llm_benchmark.py`**：核心评测脚本。负责加载模型与分词器、针对不同评测策略构建结构化 Prompt，并带有安全的原子写入检查点（Checkpoint）机制。
* **`summary_report.py`**：离线结果汇总与解析脚本。支持从检查点文件中提取模型生成的选项，并以美观的表格形式打印各任务和策略的准确率。

---

## 支持的模型与评测策略

### 1. 评测模型
* `TinyLlama/TinyLlama_v1.1`
* `Qwen/Qwen2.5-3B-Instruct`
* `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
* `Qwen/Qwen3-4B-Instruct-2507`

### 2. 评测策略
* **Baseline / PE**：0-shot 基础测试与优化提示词测试。
* **FewShot**：Few-shot 上下文学习测试。
* **CoT**：结合 1-shot 示例、`<think>` 思维链引导及结构化终结词（`Final Answer: [LETTER]`）的推理测试。
---

##  环境依赖

在运行项目前，请确保安装了以下 Python 依赖库：

```bash
```

--
##  快速使用指南

### 1. 运行评测
执行主脚本开始对指定模型和任务进行评测：
```bash
python open_llm_benchmark.py
```
> 💡 **提示**：脚本支持**断点续跑**。如果在运行途中中断，再次运行会自动读取 `benchmark_checkpoint.json` 从断点处继续。你也可以在代码中将 `FAST_TEST = True` 改为快速测试模式。

### 2. 查看汇总报告
评测完成后，运行汇总脚本以离线方式计算准确率并输出结构化对比表格：
```bash
python summary_report.py
```

---

如有任何关于 Prompt 调整或结果解析的优化需求，欢迎随时修改扩展！