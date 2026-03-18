import json

import pandas as pd
import pytest

import openjury.instruction_dataset.mt_bench_101 as mt_bench_101_dataset
from openjury.mt_bench_101.evaluate import (
    derive_mt_bench_101_pairwise_preferences,
    judge_mt_bench_101_single,
    parse_mt_bench_101_rating,
    summarize_mt_bench_101_absolute_scores,
    summarize_mt_bench_101_pairwise,
)
from openjury.utils import DummyModel


def test_load_mt_bench_101_turn_expansion(tmp_path, monkeypatch):
    dataset_path = tmp_path / "mtbench101.jsonl"
    records = [
        {
            "task": "CM",
            "id": 1,
            "history": [
                {"user": "u1", "bot": "b1"},
                {"user": "u2", "bot": "b2"},
            ],
        },
        {
            "task": "PI",
            "id": 2,
            "history": [
                {"user": "x1", "bot": "y1"},
                {"user": "x2", "bot": "y2"},
            ],
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mt_bench_101_dataset,
        "download_mt_bench_101",
        lambda local_dir=None: dataset_path,
    )

    eval_items = mt_bench_101_dataset.load_mt_bench_101()

    # CM starts at turn 2 (1 row), PI starts at turn 1 (2 rows) => total 3.
    assert len(eval_items) == 3
    cm_rows = eval_items[eval_items["task"] == "CM"]
    assert cm_rows.iloc[0]["turn_index"] == 2
    assert len(cm_rows.iloc[0]["golden_context"]) == 1


def test_parse_mt_bench_101_rating():
    assert parse_mt_bench_101_rating("Reasoning...\nRating: [[7]]") == pytest.approx(7.0)
    assert parse_mt_bench_101_rating("rating: [[10]]") == pytest.approx(10.0)
    assert parse_mt_bench_101_rating("I would rate this [[6]] overall.") == pytest.approx(6.0)
    assert (
        parse_mt_bench_101_rating("See section [3] for details...\nRating: [[6]]")
        == pytest.approx(6.0)
    )
    assert parse_mt_bench_101_rating("Rating: [[0]]") is None
    assert parse_mt_bench_101_rating("Rating: [[11]]") is None
    assert parse_mt_bench_101_rating("Rating: [6]") is None
    assert parse_mt_bench_101_rating("No rating present.") is None


def test_judge_mt_bench_101_includes_reference_block_for_mr():
    eval_items = pd.DataFrame(
        {
            "instruction_index": [0],
            "dialogue_id": [1],
            "dialogue_uid": ["MR:1"],
            "task": ["MR"],
            "ability": ["adaptability"],
            "turn_index": [2],
            "golden_context": [[{"user": "q1", "bot": "a1"}]],
            "user_message": ["q2"],
            "reference_answer": ["ref answer"],
        }
    ).set_index("instruction_index")
    completions = pd.DataFrame(
        {"instruction_index": [0], "completion": ["model answer"]}
    )

    scored = judge_mt_bench_101_single(
        judge_chat_model=DummyModel("Dummy/reasoning\nRating: [[8]]"),
        eval_items=eval_items,
        completions=completions,
        use_tqdm=False,
    )

    user_prompt = scored.iloc[0]["user_prompt"]
    assert scored.iloc[0]["score"] == pytest.approx(8.0)
    assert "The dialogue need to be judged is:" in user_prompt
    assert "The reference solution is:" in user_prompt
    assert " Human: q1" in user_prompt
    assert "Assistant: model answer" in user_prompt
    assert user_prompt.find("***") < user_prompt.find("The reference solution is:")
    assert "strictly following this format" in scored.iloc[0]["system_prompt"]


def test_mt_bench_101_aggregation_and_pairwise():
    scored_a = pd.DataFrame(
        {
            "instruction_index": [0, 1, 2],
            "dialogue_uid": ["PI:1", "PI:1", "PI:2"],
            "dialogue_id": [1, 1, 2],
            "task": ["PI", "PI", "PI"],
            "ability": ["interactivity", "interactivity", "interactivity"],
            "turn_index": [1, 2, 1],
            "score": [9.0, 2.0, 4.0],
        }
    )
    scored_b = pd.DataFrame(
        {
            "instruction_index": [0, 1, 2],
            "dialogue_uid": ["PI:1", "PI:1", "PI:2"],
            "dialogue_id": [1, 1, 2],
            "task": ["PI", "PI", "PI"],
            "ability": ["interactivity", "interactivity", "interactivity"],
            "turn_index": [1, 2, 1],
            "score": [8.0, 1.0, 6.0],
        }
    )

    absolute_a = summarize_mt_bench_101_absolute_scores(scored_turns=scored_a)
    assert absolute_a["per_task"]["PI"] == pytest.approx(3.0)
    assert absolute_a["overall"] == pytest.approx(3.0)

    pairwise_turns = derive_mt_bench_101_pairwise_preferences(scored_a, scored_b)
    summary = summarize_mt_bench_101_pairwise(pairwise_turns=pairwise_turns)

    assert summary["turn_level"]["num_battles"] == 3
    assert summary["dialogue_level"]["num_battles"] == 2
    # dialogue-level uses min scores per dialogue: one A win and one A loss
    assert summary["dialogue_level"]["winrate"] == pytest.approx(0.5)
