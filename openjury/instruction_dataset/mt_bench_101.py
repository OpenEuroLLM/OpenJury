import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from openjury.utils import data_root

MT_BENCH_101_DATA_URL = (
    "https://raw.githubusercontent.com/mtbench101/mt-bench-101/main/"
    "data/subjective/mtbench101.jsonl"
)

MT_BENCH_101_TURN2_ONLY_TASKS = {"CM", "AR", "CR", "FR", "SC", "SA"}
MT_BENCH_101_REFERENCE_TASKS = {"MR", "GR"}
MT_BENCH_101_TASK_TO_ABILITY = {
    "CM": "perceptivity",
    "AR": "perceptivity",
    "SI": "perceptivity",
    "TS": "perceptivity",
    "CC": "perceptivity",
    "CR": "adaptability",
    "FR": "adaptability",
    "SC": "adaptability",
    "SA": "adaptability",
    "MR": "adaptability",
    "GR": "adaptability",
    "IC": "interactivity",
    "PI": "interactivity",
}


def download_mt_bench_101(local_dir: Path | None = None) -> Path:
    """Download MT-Bench-101 JSONL dataset if missing and return its path."""
    if local_dir is None:
        local_dir = data_root / "mt-bench-101"

    local_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = local_dir / "data" / "subjective" / "mtbench101.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        return dataset_path

    try:
        urlretrieve(MT_BENCH_101_DATA_URL, dataset_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download MT-Bench-101 dataset from GitHub. "
            "If running in a restricted network environment, manually place the file at "
            f"{dataset_path} or point OPENJURY_DATA to a cache containing it."
        ) from exc

    return dataset_path


def load_mt_bench_101(
    n_dialogues: int | None = None,
    local_dir: Path | None = None,
) -> pd.DataFrame:
    """Load MT-Bench-101 and expand dialogues into turn-level evaluation items.

    The returned dataframe has one row per evaluated turn, using golden context.
    """
    dataset_path = download_mt_bench_101(local_dir=local_dir)

    records: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if n_dialogues is not None:
        records = records[:n_dialogues]

    rows: list[dict] = []
    for rec in records:
        task = rec.get("task")
        if task not in MT_BENCH_101_TASK_TO_ABILITY:
            raise ValueError(f"Unknown MT-Bench-101 task '{task}' in record: {rec}")

        dialogue_id = rec.get("id")
        history = rec.get("history")
        if not isinstance(history, list):
            raise ValueError(
                "Invalid MT-Bench-101 record: expected list in field 'history', "
                f"got {type(history)}"
            )

        start_turn = 2 if task in MT_BENCH_101_TURN2_ONLY_TASKS else 1
        for turn_pos, turn in enumerate(history, start=1):
            if turn_pos < start_turn:
                continue
            if not isinstance(turn, dict):
                raise ValueError(
                    "Invalid MT-Bench-101 record: each turn in 'history' must be a dict."
                )

            user_message = str(turn.get("user") or "")
            reference_answer = str(turn.get("bot") or "")
            golden_context = [
                {
                    "user": str(prev_turn.get("user") or ""),
                    "bot": str(prev_turn.get("bot") or ""),
                }
                for prev_turn in history[: turn_pos - 1]
            ]

            rows.append(
                {
                    "instruction_index": len(rows),
                    "dialogue_id": dialogue_id,
                    "dialogue_uid": f"{task}:{dialogue_id}",
                    "task": task,
                    "ability": MT_BENCH_101_TASK_TO_ABILITY[task],
                    "turn_index": turn_pos,
                    "golden_context": golden_context,
                    "user_message": user_message,
                    "reference_answer": reference_answer,
                    "requires_reference": task in MT_BENCH_101_REFERENCE_TASKS,
                    "instruction": user_message,
                }
            )

    return pd.DataFrame(rows)
