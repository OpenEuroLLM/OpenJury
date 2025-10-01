# LLM-Judge evaluation


Use-cases:
* evaluate one model easily against another on AE/AH/m-AH
* easily swap judge model
* common format for AE/AH/m-AH

For generation and LLM-judge any model available in [LangChain](https://python.langchain.com/docs/integrations/chat/])
should be usable in theory (I only tested LlamaCpp and Together, I plan to also test VLLM and OpenAI).

**Generate completions.** To generate completions, run something like this:
```bash
python generate.py \
--dataset alpaca-eval \
--model Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
--output_path results/llama-3.2-3b-q8_0.csv.zip \
--n_instructions 10
```

**Evaluate a model available in AlpacaEval against local predictions.**
To run judge LLM evaluations, run something like this:
```bash
python llmjudgeeval/evaluate.py \
--dataset alpaca-eval \
--method_A gpt4_1106_preview \
--method_B alpaca-eval-gpt-3.5-turbo.csv.zip \
--judge_model Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
--n_instructions 10
```
Note that the methods passed in `method_A` and `method_B` should be either a method existing in the
dataset used or a local file containing instructions like `alpaca-eval-gpt-3.5-turbo.csv.zip`.

To choose a model, you need to pass first the LangChain backend (LlamaCpp, ChatOpenAI, VLLM, Together...).
Here are examples with different providers:
* Together/meta-llama/Llama-3.3-70B-Instruct-Turbo 
* ChatOpenAI/gpt-5-nano
* LlamaCpp/jwiggerthale_Llama-3.2-3B-Q8_0-GGUF_llama-3.2-3b-q8_0.gguf

and an example to call llm-judge with a judge running locally on VLLM:
```bash
python llmjudgeeval/evaluate.py \
--dataset alpaca-eval \
--method_A gpt4_1106_preview \
--method_B alpaca-eval-gpt-3.5-turbo.csv.zip \
--judge_model VLLM/meta-llama/Meta-Llama-3-8B-instruct \
--n_instructions 10
```

TODOs [priority/size]:
* mAH: support using all languages at once [high/medium]
* support evaluation with input swap [medium/small]
* handle errors [medium/small]
* document options [medium/large]
* add details to example to generate and evaluate completions [medium/medium] 
* CI [high/large]
* implement CI judge option
* implement domain filter in CI (maybe pass a regexp by column?)
* cost? 

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
