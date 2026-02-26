"""Rubric I/O helpers (JSON-only for now)."""

from __future__ import annotations

import json
from pathlib import Path

from openjury.rubrics.defaults import get_rubric, register_rubric
from openjury.rubrics.schema import Criterion, Rubric


def load_rubric_from_json(path: str | Path) -> Rubric:
    """Load a rubric definition from a JSON file.

    Supports both the preferred ``"criteria"`` key and the legacy
    ``"dimensions"`` key (for backward compatibility).
    """
    data = json.loads(Path(path).read_text())
    criteria_data = data.get("criteria", data["dimensions"])
    criteria = [
        Criterion(
            name=d["name"],
            description=d["description"],
            scale_min=d.get("scale_min", 1),
            scale_max=d.get("scale_max", 10),
            weight=d.get("weight", 1.0),
            score_references=d.get("score_references", {}),
        )
        for d in criteria_data
    ]
    return Rubric(
        name=data["name"],
        criteria=criteria,
        description=data.get("description", ""),
    )


def register_rubric_from_json(path: str | Path) -> Rubric:
    """Load a rubric JSON file and register it in the runtime registry."""
    rubric = load_rubric_from_json(path)
    register_rubric(rubric)
    return rubric


def load_rubric_from_file(path: str | Path) -> Rubric:
    """Load a rubric from a file path.

    Currently only JSON files are supported. This function exists to keep the
    call site API format-agnostic for future YAML support.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported rubric file format '{path.suffix}'. "
            "Only .json is supported in this PR."
        )
    return load_rubric_from_json(path)


def register_rubric_from_file(path: str | Path) -> Rubric:
    """Load a rubric file and register it (JSON-only for now)."""
    rubric = load_rubric_from_file(path)
    register_rubric(rubric)
    return rubric


def resolve_rubric(
    rubric_name: str = "default",
    rubric_file: str | Path | None = None,
) -> Rubric:
    """Resolve rubric by name or file path (file path takes precedence).

    Currently ``rubric_file`` supports JSON only.
    """
    if rubric_file is not None:
        return register_rubric_from_file(rubric_file)
    return get_rubric(rubric_name)
