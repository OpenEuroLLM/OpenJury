# 🏛️ OpenJury: LLM Evaluation with Swappable Judges

OpenJury makes it easy to benchmark language models against each other while giving you complete control over the evaluation process. 
Whether you're comparing proprietary models or testing your own fine-tuned creations, OpenJury lets you choose your judge.

## ✨ Key Features

🎯 **Flexible Benchmarking** – Evaluate models on `Alpaca-Eval`, `Arena-Hard`, `m-Arena-Hard` and others

🔄 **Swappable Judges** – Switch between self-hosted (`vLLM`) or remote judges (`OpenAI`, `Together AI`, `OpenRouter`)

🌍 **Multilingual Support** – Test models across multiple languages with m-Arena-Hard

🛠️ **Provider Agnostic** – Works with any model available in [LangChain](https://python.langchain.com/docs/integrations/chat/)

Compared to other libraries, here is a breakdown of features:

| Framework | MT-Bench | AlpacaEval | Arena-Hard | M-Arena-Hard | Tuned Judge configuration | Support vLLM Judges |
|-----------|----------|------------|------------|--------------|---------------------------|---------------------|
| **FastChat** | ✅  | ❌  | ❌  | ❌  | ❌    | ❌                        |
| **AlpacaEval** | ❌  | ✅  | ❌  | ❌  | ❌   | ❌                                             |
| **Arena-Hard-Auto** | ❌  | ❌  | ✅  | ❌  | ❌    | ❌                                            |
| **Lighteval** | ✅  | ❌  | ❌  | ❌  | ❌         | ❌                                       |
| **Evalchemy** | ✅  | ✅  | ❌  | ❌  | ❌     | ❌                                           |
| **OpenJury** | 🔜  | ✅  | ✅  | ✅  | ✅     | ✅                                          |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync 
uv sync --extra vllm   # Optional: install vLLM support
```

### Basic Evaluation

Compare two models head-to-head:

```bash
python openjury/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --model_B gpt4_1106_preview \
  --judge_model Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --n_instructions 10 
```

**What happens here?**
- Generates completions for `model_A` if not already cached
- Compares against existing `model_B` completions from Alpaca-Eval database
- Uses your chosen judge model to evaluate the results

## 🎨 Model Specification

Models are specified using the format: `{LangChain Backend}/{Model Path}`

**Examples:**

```bash
Together/meta-llama/Llama-3.3-70B-Instruct-Turbo
ChatOpenAI/gpt-4o
LlamaCpp/jwiggerthale_Llama-3.2-3B-Q8_0-GGUF_llama-3.2-3b-q8_0.gguf
VLLM/MultiSynt/nemotron-cc-german-9b
OpenRouter/deepseek/deepseek-chat-v3.1
```

For instance, to run everything locally with vLLM:

```bash
python openjury/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model_B VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --n_instructions 10 
```

**Note:** Ensure you have the required LangChain dependencies installed for your chosen provider.

## 📊 Supported Datasets

| Dataset | Description                                                                                    |
|---------|------------------------------------------------------------------------------------------------|
| `alpaca-eval` | General instruction-following benchmark                                                        |
| `arena-hard` | More challenging evaluation suite                                                              |
| `m-arena-hard` | Translated version of Arena-Hard in 23 languages                                               |
| `m-arena-hard-{lang}` | Language-specific variants (e.g., `ar`, `cs`, `de`)                                            |
| `m-arena-hard-EU` | All EU languages combined                                                                      |
| `{lang}-contexts` | Fluency evaluation for pretrained models (`finnish`, `french`, `german`, `spanish`, `swedish`) |

### Offline Setup (Slurm/Air-Gapped Environments)

Pre-download all datasets before running jobs:

```bash
python -c "from openjury.utils import download_all; download_all()"  # Download all datasets (optional)
```

Datasets are stored in:
- `$OPENJURY_EVAL_DATA` (if set)
- `~/openjury-eval-data/` (default)

## 🤝 Contributing

We welcome contributions! Whether it's bug fixes, new features, or additional benchmark support, feel free to open an issue or submit a pull request.

## Citation

If you use this work in your research, please cite the following paper.

```bibtex
@inproceedings{
  salinas2025tuning,
  title={Tuning {LLM} Judge Design Decisions for 1/1000 of the Cost},
  author={David Salinas and Omar Swelam and Frank Hutter},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=cve4NOiyVp}
}
```

The judge configurations are the best one found in this paper and a lot of code is reused in this package.

---