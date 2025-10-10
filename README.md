# OpenJury: LLM Evaluation with Swappable Judges

The main use-cases of this packages are:
* evaluating one model easily against another on `Alpaca-Eval`, `Arena-Hard`, `m-Arena-Hard` or other benchmarks
* easily swap judge model among self-hosted options with `vLLM` or remote options with `Together` or `OpenAI`

For generation and LLM-judge any model available in [LangChain](https://python.langchain.com/docs/integrations/chat/]) should be usable in theory, so far LlamaCpp, vLLM, 
Together and OpenAI have been tested which should already cover a wide set of use-cases.


**Installation instructions.**

```bash
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync 
uv sync --extra vllm   # if you need vllm
export OPENJURY_EVAL_DATA=~/openjury-eval-data/  # where data is downloaded
python -c "from openjury.utils import download_all; download_all()"  # if you need to download all datasets at once
```


**Evaluate a model.** To evaluate a model, you can run the following:
```bash
python openjury/generate_and_evaluate.py \
--dataset alpaca-eval \
--model_A Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
--model_B gpt4_1106_preview \
--judge_model Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
--n_instructions 10 
```

This will compare the completions of `Together/meta-llama/Llama-3.3-70B-Instruct-Turbo` with `gpt4_1106_preview` 
by the judge `Together/meta-llama/Llama-3.3-70B-Instruct-Turbo`. Completions for `model_A` will be generated if they 
are not present already and will then compare with completions
of `gpt4_1106_preview` available in alpaca-eval.

To choose a model, you need to pass first the LangChain backend (LlamaCpp, ChatOpenAI, VLLM, Together...).
Here are examples with different providers:
* `Together/meta-llama/Llama-3.3-70B-Instruct-Turbo` 
* `ChatOpenAI/gpt-5-nano`
* `LlamaCpp/jwiggerthale_Llama-3.2-3B-Q8_0-GGUF_llama-3.2-3b-q8_0.gguf`
* `VLLM/MultiSynt/nemotron-cc-german-9b`

```bash
python openjury/generate_and_evaluate.py \
--dataset alpaca-eval \
--model_A VLLM/Qwen/Qwen2.5-0.5B-Instruct \
--model_B VLLM/Qwen/Qwen2.5-1.5B-Instruct \
--judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
--n_instructions 10 
```

You should make sure that you have the extra-dependencies of Langchain installed.

**Dataset supported.** The following datasets are supported:
* "alpaca-eval"
* "arena-hard"
* "m-arena-hard"
* "m-arena-hard-{lang}" where lang is one of the language supported by m-Arena-Hard (e.g. "ar", "cs", "de") or "EU" to 
run on all EU languages
* {lang}-contexts where lang is one of "finnish", "french", "german", "spanish", "swedish". A setup to evaluated the 
fluency of pretrained models. This datasets consists in sentences that needs to be completed and the completion are evaluated by an LLM-judge.

If you are running in a Slurm setup without internet access on compute nodes, you may want to pre-download all datasets
locally by running:

```
python -c "from openjury.utils import download_all; download_all()"
```

The datasets will be downloaded under `$OPENJURY_EVAL_DATA` if the environment variable is specified and 
`~/openjury-eval-data/` otherwise. 

