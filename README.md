# 🏛️ OpenJury: LLM Evaluation with Swappable Judges

OpenJury makes it easy to benchmark language models against each other while giving you complete control over the evaluation process. 
Whether you're comparing proprietary models or testing your own fine-tuned creations, OpenJury lets you choose your judge.

## ✨ Key Features

🎯 **Flexible Benchmarking** – Evaluate models on `Alpaca-Eval`, `Arena-Hard`, `m-Arena-Hard` and others

🔄 **Swappable Judges** – Switch between self-hosted (`vLLM`) or remote judges (`OpenAI`, `Together AI`, `OpenRouter`)

🌍 **Multilingual Support** – Test models across multiple languages with m-Arena-Hard

🛠️ **Provider Agnostic** – Works with any model available in [LangChain](https://python.langchain.com/docs/integrations/chat/)

Compared to other libraries, here is a breakdown of features:

| Framework | MT-Bench | AlpacaEval | Arena-Hard | M-Arena-Hard | Tuned judge configuration | Support vLLM Judges |
|-----------|----------|------------|------------|--------------|---------------------------|---------------------|
| **FastChat** | ✅  | ❌  | ❌  | ❌  | ❌                         | ❌                        |
| **AlpacaEval** | ❌  | ✅  | ❌  | ❌  | ❌                         | ❌                                             |
| **Arena-Hard-Auto** | ❌  | ❌  | ✅  | ❌  | ❌                         | ❌                                            |
| **Lighteval** | ✅  | ❌  | ❌  | ❌  | ❌                         | ❌                                       |
| **Evalchemy** | ✅  | ✅  | ❌  | ❌  | ❌                         | ❌                                           |
| **OpenJury** | 🔜  | ✅  | ✅  | ✅  | ✅                         | ✅                                          |

The table has been done on Oct 2025, in case some libraries implemented missing features, please open an issue 
or send a PR, we will be happy to update the information.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync 
uv sync --extra vllm      # Optional: install vLLM support
uv sync --extra llamacpp   # Optional: install LlamaCpp support
```

### Automatic Environment (direnv)

We use `direnv` to automatically manage the virtual environment and environment variables.

1. **Install direnv:**
```bash
curl -sfL https://direnv.net/install.sh | bash
```

2. **Setup Hook:**
```bash
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc # or ~/.bashrc depending on which shell you use
```

3. **Configure:** Create a `.envrc` file in the project root:
```bash
dotenv_if_exists
```

4. **Allow:** Run `direnv allow` to authorize the setup.

5. **Env variables** Add env variables like `OPEN_JURY_EVAL_DATA` to `.env` file.

### Basic Evaluation

Compare two models head-to-head:

```bash
python openjury/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A gpt4_1106_preview \
  --model_B VLLM/utter-project/EuroLLM-9B-Instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 10 
```

**What happens here?**
- Use completions available for `gpt4_1106_preview` in Alpaca-Eval dataset
- Generates completions for `model_B` if not already cached on `vLLM`
- Compares two models using `deepseek-chat-v3.1` which the cheapest option available on `OpenRouter` 

It will then display the results of the battles:

```bash
============================================================
                  🏆 MODEL BATTLE RESULTS 🏆                  
📊 Dataset: alpaca-eval
🤖 Competitors: Model A: gpt4_1106_preview vs Model B: VLLM/utter-project/EuroLLM-9B-Instruct
⚖️ Judge: OpenRouter/deepseek/deepseek-chat-v3.1
📈 Results Summary:
   Total Battles: 10
   Win Rate (A): 30.0%
   ✅ Wins:   3
   ❌ Losses: 6
   🤝 Ties:   1
============================================================
```

## 🎨 Model Specification

Models are specified using the format: `{LangChain Backend}/{Model Path}`

**Examples:**

```bash
Together/meta-llama/Llama-3.3-70B-Instruct-Turbo
ChatOpenAI/gpt-4o
LlamaCpp/jwiggerthale_Llama-3.2-3B-Q8_0-GGUF_llama-3.2-3b-q8_0.gguf
VLLM/utter-project/EuroLLM-9B-Instruct
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

### Running locally with LlamaCpp

LlamaCpp lets you run GGUF models locally on CPU, which is useful for testing your setup without needing a GPU or API keys.

**Install the LlamaCpp extra:**

```bash
uv sync --extra llamacpp
```

**Download GGUF models** using `huggingface-cli` (included via `huggingface-hub`):

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir ./models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf --local-dir ./models
```

The `LlamaCpp` provider expects a **file path** to a `.gguf` model after the `LlamaCpp/` prefix.
For absolute paths, this results in a double slash (e.g., `LlamaCpp//home/user/models/model.gguf`).

**Mixed example** — local LlamaCpp model with a remote judge:

```bash
uv run python openjury/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model_B OpenRouter/qwen/qwen-2.5-7b-instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 10 --max_out_tokens_models 16384
```

**Fully local example** — no API keys required (useful for verifying your setup):

```bash
uv run python openjury/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model_B LlamaCpp/./models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --judge_model LlamaCpp/./models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --n_instructions 5 --max_out_tokens_models 16384
```

**Note:** Ensure you have the required LangChain dependencies installed for your chosen provider.
If you use remote endpoint, you would have to set your credentials.

## 📊 Supported Datasets

| Dataset               | Description                                                                                    |
|-----------------------|------------------------------------------------------------------------------------------------|
| `alpaca-eval`         | General instruction-following benchmark                                                        |
| `arena-hard`          | More challenging evaluation suite                                                              |
| `m-arena-hard`        | Translated version of Arena-Hard in 23 languages                                               |
| `m-arena-hard-{lang}` | Language-specific variants (e.g., `ar`, `cs`, `de`)                                            |
| `m-arena-hard-EU`     | All EU languages combined                                                                      |
| `fluency-{lang}`      | Fluency evaluation for pretrained models (`finnish`, `french`, `german`, `spanish`, `swedish`) |

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

The judge configurations was tuned in this paper and a lot of code is reused in this package.

---