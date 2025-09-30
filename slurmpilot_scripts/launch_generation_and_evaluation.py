from pathlib import Path

from slurmpilot import SlurmPilot, JobCreationInfo, unify

cluster = "kislurm"
slurm = SlurmPilot(clusters=[cluster])
baseline = "VLLM/Qwen/Qwen2.5-7B"

models = [
    "VLLM/MultiSynt/nemotron-cc-spanish-tower9b",
    # "VLLM/MultiSynt/nemotron-cc-spanish-run1-tower72b",
    "VLLM/MultiSynt/nemotron-cc-translated-100B-opus-mt-spa",
    "VLLM/HPLT/hplt2c_spa_checkpoints",
]
job_info = JobCreationInfo(
    cluster=cluster,
    partition="alldlc2_gpu-h200",
    jobname=unify(f"oellmjudge/spanish-eval", method="date"),
    entrypoint="generate_and_evaluate.py",
    python_binary="/work/dlclarge1/salinasd-llm-judge/llm-judge-eval/.venv/bin/python",
    python_args=[
        {
            "dataset": "spanish-contexts",
            "generation_model_A": baseline,
            "generation_model_B": model,
            "judge_model": "VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
            "n_instructions": 50,
            # "ignore_cache": None,
        }
        for model in models
    ],
    src_dir=str(Path(__file__).parent.parent / "llmjudgeeval/"),
    python_libraries=[str(Path(__file__).parent.parent / "data/")],
    n_cpus=1,
    max_runtime_minutes=60,
    env={"HF_HUB_OFFLINE": "1"},
)

# Launch the job
job_id = slurm.schedule_job(job_info)
print(f"Job {job_id} scheduled on {job_info.cluster}")
# slurm.wait_completion(job_info.jobname)
