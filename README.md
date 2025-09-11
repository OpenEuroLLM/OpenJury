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
--model_provider LlamaCpp \
--model_kwargs model_path=jwiggerthale_Llama-3.2-3B-Q8_0-GGUF_llama-3.2-3b-q8_0.gguf max_retries=3 \
--output_path results/llama-3.2-3b-q8_0.csv.zip \
--n_instructions 10
```

**Evaluate a model available in AlpacaEval against local predictions.**
To run judge LLM evaluations, run something like this:
```bash
python llm-judge-eval/evaluate.py \
--dataset alpaca-eval \
--method_A gpt4_1106_preview \
--method_B alpaca-eval-gpt-3.5-turbo.csv.zip \
--judge_provider Together \
--judge_model "meta-llama/Llama-3.3-70B-Instruct-Turbo" \
--n_instructions 100
```

TODOs:
* support m-arena-hard [high/large]
* support evaluation with input swap [medium/small]
* test openai judge [medium/small]
* test vLLM judge [medium/small]
* handle errors [medium/small]
* CLI launcher [medium/large]
* document options [medium/large]
* add example to generate and evaluate completions [medium/medium] 
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