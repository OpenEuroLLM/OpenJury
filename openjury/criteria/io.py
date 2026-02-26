"""Criteria-set I/O helpers (JSON-only for now)."""

from __future__ import annotations

import json
from pathlib import Path

from openjury.criteria.defaults import get_criteria, register_criteria
from openjury.criteria.schema import Criterion, Criteria


def load_criteria_from_json(path: str | Path) -> Criteria:
    """Load a criteria-set definition from a JSON file.

    Expects the preferred ``"criteria"`` key.
    """
    data = json.loads(Path(path).read_text())
    criteria_data = data["criteria"]
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
    return Criteria(
        name=data["name"],
        criteria=criteria,
        description=data.get("description", ""),
    )


def register_criteria_from_json(path: str | Path) -> Criteria:
    """Load a criteria-set JSON file and register it in the runtime registry."""
    criteria = load_criteria_from_json(path)
    register_criteria(criteria)
    return criteria


def load_criteria_from_file(path: str | Path) -> Criteria:
    """Load a criteria set from a file path.

    Currently only JSON files are supported. This function exists to keep the
    call site API format-agnostic for future YAML support.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported criteria file format '{path.suffix}'. "
            "Only .json is supported in this PR."
        )
    return load_criteria_from_json(path)


def register_criteria_from_file(path: str | Path) -> Criteria:
    """Load a criteria-set file and register it (JSON-only for now)."""
    criteria = load_criteria_from_file(path)
    register_criteria(criteria)
    return criteria


def resolve_criteria(
    criteria_name: str = "default",
    criteria_file: str | Path | None = None,
) -> Criteria:
    """Resolve a criteria set by name or file path (file path takes precedence).

    Currently ``criteria_file`` supports JSON only.
    """
    if criteria_file is not None:
        return register_criteria_from_file(criteria_file)
    return get_criteria(criteria_name)
