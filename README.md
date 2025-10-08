# LLM-Judge evaluation

The main use-cases of this packages are:
* evaluating one model easily against another on `Alpaca-Eval`, `Arena-Hard`, `m-Arena-Hard` or other benchmarks
* easily swap judge model among self-hosted options with `vLLM` or remote options with `Together` or `OpenAI`

For generation and LLM-judge any model available in [LangChain](https://python.langchain.com/docs/integrations/chat/]) should be usable in theory, so far LlamaCpp, vLLM, 
Together and OpenAI have been tested which should already cover a wide set of use-cases.


**Installation instructions.**

```bash
git clone https://github.com/geoalgo/llm-judge-eval
cd llm-judge-eval
uv sync 
uv sync --extra vllm   # if you need vllm
export LLM_JUDGE_EVAL_DATA=~/llm-judge-eval-data/  # where data is downloaded
python -c "from llmjudgeeval.utils import download_all; download_all()"  # if you need to download all datasets at once
```


**Evaluate a model.** 

To evaluate a model, run something like this:
```bash
python llmjudgeeval/generate_and_evaluate.py \
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
python llmjudgeeval/generate_and_evaluate.py \
--dataset alpaca-eval \
--model_A VLLM/Qwen/Qwen2.5-0.5B-Instruct \
--model_B VLLM/Qwen/Qwen2.5-1.5B-Instruct \
--judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
--n_instructions 10 
```

You should make sure that you have the extra-dependencies of Langchain installed.

**Dataset supported.**

The following datasets are supported:
* "alpaca-eval"
* "arena-hard"
* "m-arena-hard"
* "m-arena-hard-{lang}" where lang is one of the language supported by m-Arena-Hard (e.g. "ar", "cs", "de") or "EU" to 
run on all EU languages
* {lang}-contexts where lang is one of "finnish", "french", "german", "spanish", "swedish"

If you are running in a Slurm setup without internet access on compute nodes, you may want to pre-download all datasets
locally by running:

```
python -c "from llmjudgeeval.utils import download_all; download_all()"
```

The datasets will be downloaded under `$LLM_JUDGE_EVAL_DATA` if the environment variable is specified and 
`~/llm-judge-eval-data/` otherwise. 

TODOs:
* support evaluation with input swap 
* handle errors
* CI [high/large]
* implement CI judge option
* implement domain filter in CI (maybe pass a regexp by column?)
* report cost? 

Done:
* support alpaca-eval
* support arena-hard
* test together judge
* local env variable to set paths
* tqdm callback with batch
* support loading local completions
* support dumping outputs [medium/small]
* test LlamaCpp [medium/small]
* test openai judge [medium/small]
* test vLLM judge [medium/small]
* CLI launcher [medium/large]
* put contexts in HF dataset [high/small]
* mAH: instruction loader [DONE]
* mAH: generate instructions for two models [DONE] 
* mAH: make comparison [DONE]
* mAH: support using all languages at once [high/medium]
* unit-test
* add details to example to generate and evaluate completions
* installation instructions