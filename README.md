# LLM-Judge evaluation


Use-cases:
* evaluate instruction-tuned vs base model
* evaluate one model easily against another on AE/AH/m-AH
* easily swap judge model
* common format for AE/AH/m-AH

```bash
python llm-judge-eval/main.py \
--dataset alpaca-eval \
--method_A gpt4_1106_preview \
--method_B alpaca-eval-gpt-3.5-turbo.csv.zip \
--judge_provider Together \
--judge_model "meta-llama/Llama-3.3-70B-Instruct-Turbo" \
--n_instructions 100
```

```bash
python llm-judge-eval/main.py \
--dataset m-arena-hard \
--domain fr \
--method_A gpt4_1106_preview \
--method_B alpaca-eval-gpt-3.5-turbo.csv.zip \
--judge_provider Together \
--judge_model "meta-llama/Llama-3.3-70B-Instruct-Turbo" \
--n_instructions 100
```


TODOs:
* support m-arena-hard [high/large]
* support dumping outputs [medium/small]
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

