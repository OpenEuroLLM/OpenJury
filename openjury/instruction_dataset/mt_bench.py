import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from openjury.utils import data_root

def _read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    raise ValueError(f"Unsupported MT-Bench file format: {path}")


def _try_resolve_mt_bench_paths(
    root: Path,
) -> tuple[Path | None, Path | None]:
    """Find question.jsonl and reference_answer/*.jsonl under root."""
    question_candidates = [
        root / "data" / "mt_bench" / "question.jsonl",
        root / "data" / "mt_bench" / "questions.jsonl",
    ]
    # FastChat stores reference answers inside a *directory* (one file per model).
    ref_dir_candidates = [
        root / "data" / "mt_bench" / "reference_answer",
    ]

    question_path = next((p for p in question_candidates if p.exists()), None)
    if question_path is None:
        for p in root.rglob("question.jsonl"):
            question_path = p
            break
    if question_path is None:
        for p in root.rglob("questions.jsonl"):
            question_path = p
            break

    ref_path: Path | None = None
    for d in ref_dir_candidates:
        if d.is_dir():
            gpt4 = d / "gpt-4.jsonl"
            if gpt4.exists():
                ref_path = gpt4
            else:
                jsonl_files = sorted(d.glob("*.jsonl"))
                if jsonl_files:
                    ref_path = jsonl_files[0]
            break

    if ref_path is None:
        for name in ("gpt-4.jsonl", "reference_answer.jsonl", "reference_answers.jsonl"):
            for p in root.rglob(name):
                ref_path = p
                break
            if ref_path is not None:
                break

    return question_path, ref_path


def _extract_ref_turns(rec: dict) -> list[str] | None:
    """Extract reference answer turns from either model-answer or flat format.

    FastChat reference_answer/gpt-4.jsonl uses:
        {"choices": [{"turns": ["ans1", "ans2"]}]}
    while question.jsonl inlines:
        {"reference": ["hint1", "hint2"]}
    """
    choices = rec.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        turns = choices[0].get("turns")
        if isinstance(turns, list):
            return turns
    turns = rec.get("turns")
    if isinstance(turns, list):
        return turns
    return None


def load_mt_bench() -> pd.DataFrame:
    """Load MT-Bench questions (and reference answers when available).

    Downloads the MT-Bench HuggingFace space snapshot to `$OPENJURY_DATA/mt-bench/`
    (or `~/openjury-data/mt-bench/`) and returns a DataFrame with at least:
        - instruction_index (question id)
        - category
        - turn_1, turn_2
        - reference_turn_1, reference_turn_2 (may be missing/NaN)
    """
    local_dir = data_root / "mt-bench"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create MT-Bench cache directory at {local_dir}. "
            "Set environment variable OPENJURY_DATA to a writable location."
        ) from e

    question_path, ref_path = _try_resolve_mt_bench_paths(local_dir)
    if question_path is None:
        try:
            snapshot_download(
                repo_id="lmsys/mt-bench",
                repo_type="space",
                allow_patterns=[
                    "data/mt_bench/question.jsonl",
                    "data/mt_bench/reference_answer/*",
                ],
                local_dir=local_dir,
                force_download=False,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to download MT-Bench questions from HuggingFace space "
                "'lmsys/mt-bench'. If you're in an offline / restricted-network "
                "environment, pre-download the space snapshot and place the "
                "questions file under "
                f"{local_dir}/data/mt_bench/question.jsonl (and optionally "
                "reference_answer/gpt-4.jsonl), or set OPENJURY_DATA to point "
                "to that directory."
            ) from e
        question_path, ref_path = _try_resolve_mt_bench_paths(local_dir)

    if question_path is None:
        raise FileNotFoundError(
            "Could not locate MT-Bench questions after download. "
            f"Searched under {local_dir}. "
            "Expected a file like 'data/mt_bench/question.jsonl'."
        )

    questions = _read_json_or_jsonl(question_path)

    # --- Load reference answers from the separate reference file (gpt-4.jsonl) ---
    ref_by_id: dict[int | str, list[str]] = {}
    if ref_path is not None:
        for rec in _read_json_or_jsonl(ref_path):
            qid = rec.get("question_id", rec.get("id"))
            if qid is None:
                continue
            turns = _extract_ref_turns(rec)
            if turns is not None:
                ref_by_id[qid] = turns

    rows = []
    for rec in questions:
        qid_raw = rec.get("question_id", rec.get("id"))
        if qid_raw is None:
            raise ValueError(
                f"MT-Bench question record missing question_id/id: keys={list(rec.keys())}"
            )
        try:
            qid = int(qid_raw)
        except Exception:
            qid = qid_raw

        category = rec.get("category")
        turns = rec.get("turns")
        if isinstance(turns, list):
            turn_1 = turns[0] if len(turns) > 0 else None
            turn_2 = turns[1] if len(turns) > 1 else None
        else:
            turn_1 = rec.get("turn_1", rec.get("instruction"))
            turn_2 = rec.get("turn_2")

        # Prefer the separate gpt-4 reference file; fall back to the inline
        # "reference" field embedded in question.jsonl (short hints).
        ref_turns = ref_by_id.get(qid_raw) or ref_by_id.get(qid)
        if ref_turns is None:
            inline_ref = rec.get("reference")
            if isinstance(inline_ref, list):
                ref_turns = inline_ref

        ref_turn_1 = ref_turns[0] if isinstance(ref_turns, list) and len(ref_turns) > 0 else None
        ref_turn_2 = ref_turns[1] if isinstance(ref_turns, list) and len(ref_turns) > 1 else None

        rows.append(
            {
                "instruction_index": qid,
                "category": category,
                "turn_1": turn_1,
                "turn_2": turn_2,
                "reference_turn_1": ref_turn_1,
                "reference_turn_2": ref_turn_2,
                "instruction": turn_1,
            }
        )

    df = pd.DataFrame(rows)
    return df

