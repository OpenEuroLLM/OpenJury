"""Rubric JSON I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from openjury._logging import logger
except Exception:  # pragma: no cover
    logger = None

from openjury.rubrics.defaults import get_rubric, register_rubric
from openjury.rubrics.schema import Rubric, RubricDimension


def load_rubric_from_json(path: str | Path) -> Rubric:
    """Load a rubric definition from a JSON file."""
    data = json.loads(Path(path).read_text())
    dimensions = [
        RubricDimension(
            name=d["name"],
            description=d["description"],
            scale_min=d.get("scale_min", 1),
            scale_max=d.get("scale_max", 10),
            weight=d.get("weight", 1.0),
            score_references=d.get("score_references", {}),
        )
        for d in data["dimensions"]
    ]
    return Rubric(
        name=data["name"],
        dimensions=dimensions,
        description=data.get("description", ""),
    )


def register_rubric_from_json(path: str | Path) -> Rubric:
    """Load a rubric JSON file and register it in the runtime registry."""
    rubric = load_rubric_from_json(path)
    register_rubric(rubric)
    if logger is not None:
        logger.info("Registered rubric from JSON: %s (name=%s)", path, rubric.name)
    return rubric


def resolve_rubric(
    rubric_name: str = "default",
    rubric_json: str | Path | None = None,
) -> Rubric:
    """Resolve rubric by name or JSON path (JSON path takes precedence)."""
    if rubric_json is not None:
        return register_rubric_from_json(rubric_json)
    return get_rubric(rubric_name)
