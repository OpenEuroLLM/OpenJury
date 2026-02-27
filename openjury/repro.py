"""Reproducibility metadata helpers.

Writes a compact, versioned run metadata file that captures enough context to
reproduce a run (args, dependency versions, git state, environment and
produced artifacts list).
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

METADATA_FILENAME = "run-metadata.v1.json"
METADATA_SCHEMA_VERSION = "openjury-run-metadata/v1"
_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary objects into JSON-safe values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # JSON standard does not support NaN/Inf; encode as null.
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    # numpy / pandas scalars usually expose .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass
    return str(value)


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"))


def _extract_dist_name(requirement_spec: str) -> str | None:
    match = _REQUIREMENT_NAME_RE.match(requirement_spec or "")
    if not match:
        return None
    return match.group(1)


def _dependency_names_from_pyproject(repo_root: Path) -> list[str]:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return []

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return []

    project = data.get("project", {})
    names: set[str] = set()

    for spec in project.get("dependencies", []) or []:
        dist = _extract_dist_name(spec)
        if dist:
            names.add(dist)

    optional = project.get("optional-dependencies", {}) or {}
    for specs in optional.values():
        for spec in specs or []:
            dist = _extract_dist_name(spec)
            if dist:
                names.add(dist)

    return sorted(names)


def _project_dependency_names(start_path: Path) -> list[str]:
    names: set[str] = set()

    # Prefer installed project metadata when available.
    try:
        dist = importlib_metadata.distribution("llm-judge-eval")
        for req in dist.requires or []:
            dep = _extract_dist_name(req)
            if dep:
                names.add(dep)
    except Exception:
        pass

    if names:
        return sorted(names)

    # Fallback: parse pyproject dependencies from repo root.
    repo_root = _run_git(["rev-parse", "--show-toplevel"], cwd=start_path)
    if repo_root is None:
        return []
    return _dependency_names_from_pyproject(Path(repo_root))


def _get_dependency_versions(
    dependencies: list[str] | None = None,
    start_path: Path | None = None,
) -> dict[str, str | None]:
    dep_names = dependencies or _project_dependency_names(start_path or Path.cwd())
    versions: dict[str, str | None] = {}
    for dist_name in dep_names:
        try:
            versions[dist_name] = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            versions[dist_name] = None
        except Exception:
            versions[dist_name] = None
    return versions


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _get_git_info(start_path: Path) -> dict[str, Any]:
    repo_root = _run_git(["rev-parse", "--show-toplevel"], cwd=start_path)
    if repo_root is None:
        return {
            "repo_root": None,
            "branch": None,
            "commit": None,
        }

    root = Path(repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    status = _run_git(["status", "--porcelain"], cwd=root)
    return {
        "repo_root": str(root),
        "branch": branch,
        "commit": commit,
    }


def _collect_artifacts(output_dir: Path, metadata_filename: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == metadata_filename:
            continue
        rel = path.relative_to(output_dir)
        artifacts.append(
            {
                "path": str(rel),
                "size_bytes": path.stat().st_size,
            }
        )
    return artifacts


def _build_input_summary(input_payloads: dict[str, Any] | None) -> dict[str, Any]:
    if not input_payloads:
        return {}
    summary: dict[str, Any] = {}
    for key, payload in input_payloads.items():
        normalized = _to_jsonable(payload)
        count = len(normalized) if hasattr(normalized, "__len__") else None
        summary[key] = {
            "count": count,
        }
    return summary


def _compact_results(results: dict[str, Any] | None) -> dict[str, Any]:
    """Compact result payload for metadata storage.

    Keep summary stats but avoid embedding large per-sample arrays.
    """
    payload = _to_jsonable(results or {})
    if not isinstance(payload, dict):
        return {"value": payload}

    if "preferences" in payload:
        prefs = payload.pop("preferences")
        count = len(prefs) if hasattr(prefs, "__len__") else None
        payload["preferences_count"] = count
    return payload


def write_run_metadata(
    *,
    output_dir: str | Path,
    entrypoint: str,
    run: dict[str, Any],
    cli_args: dict[str, Any] | None = None,
    results: dict[str, Any] | None = None,
    input_payloads: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
    started_at_utc: datetime | None = None,
    metadata_filename: str = METADATA_FILENAME,
) -> Path:
    """Write run metadata JSON and return the output path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    finished = _utc_now()
    started = started_at_utc or finished
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    duration_sec = max(0.0, (finished - started).total_seconds())

    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "timestamps": {
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_sec": duration_sec,
        },
        "entrypoint": entrypoint,
        "command": {
            "argv": _to_jsonable(sys.argv),
            "cwd": str(Path.cwd()),
        },
        "run": _to_jsonable(run),
        "cli_args": _to_jsonable(cli_args or {}),
        "results": _compact_results(results),
        "inputs": _build_input_summary(input_payloads),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "user": os.getenv("USER", None),
        },
        "dependencies": _get_dependency_versions(
            start_path=Path(__file__).resolve().parent
        ),
        "git": _get_git_info(start_path=Path(__file__).resolve().parent),
    }
    if extras:
        metadata["extras"] = _to_jsonable(extras)

    metadata["artifacts"] = _collect_artifacts(
        output_path, metadata_filename=metadata_filename
    )

    metadata_path = output_path / metadata_filename
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2, allow_nan=False)
    return metadata_path
