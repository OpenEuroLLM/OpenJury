"""Criteria-set I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from openjury.criteria.defaults import CRITERIA_BY_NAME
from openjury.criteria.schema import Criterion, criteria_from_dict


def _load_criteria_data_from_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Criteria JSON must define a mapping at the top level.")
    return data


def _load_criteria_data_from_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Criteria YAML must define a mapping at the top level.")
    return data


def load_criteria_from_json(path: str | Path) -> list[Criterion]:
    """Load criteria from a JSON file."""
    return criteria_from_dict(_load_criteria_data_from_json(path))


def load_criteria_from_yaml(path: str | Path) -> list[Criterion]:
    """Load criteria from a YAML file."""
    return criteria_from_dict(_load_criteria_data_from_yaml(path))


def load_criteria_from_file(path: str | Path) -> list[Criterion]:
    """Load criteria from a file path.

    Supported formats:
    - ``.json``
    - ``.yaml``
    - ``.yml``
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_criteria_from_json(path)
    if suffix in {".yaml", ".yml"}:
        return load_criteria_from_yaml(path)
    raise ValueError(
        f"Unsupported criteria file format '{path.suffix}'. "
        "Use .json, .yaml, or .yml."
    )


def resolve_criteria(
    criteria_name: str = "default",
    criteria_file: str | Path | None = None,
) -> tuple[str, list[Criterion]]:
    """Resolve criteria by name or file path (file path takes precedence)."""
    if criteria_file is not None:
        path = Path(criteria_file)
        if path.suffix.lower() == ".json":
            data = _load_criteria_data_from_json(path)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = _load_criteria_data_from_yaml(path)
        else:
            raise ValueError(
                f"Unsupported criteria file format '{path.suffix}'. "
                "Use .json, .yaml, or .yml."
            )
        resolved_name = data.get("name", path.stem)
        return resolved_name, criteria_from_dict(data)
    if criteria_name not in CRITERIA_BY_NAME:
        available = ", ".join(sorted(CRITERIA_BY_NAME.keys()))
        raise KeyError(f"Unknown criteria '{criteria_name}'. Available: {available}")
    return criteria_name, CRITERIA_BY_NAME[criteria_name]
