"""Reproducibility metadata helpers.

Writes a compact, versioned run metadata file that captures enough context to
reproduce a run (inputs fingerprint, args, dependency versions, git state,
artifacts and hashes).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

METADATA_FILENAME = "run-metadata.v1.json"
METADATA_SCHEMA_VERSION = "openjury-run-metadata/v1"

_DEFAULT_DEPENDENCIES: list[tuple[str, str]] = [
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_community", "langchain-community"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_together", "langchain-together"),
    ("vllm", "vllm"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit_learn", "scikit-learn"),
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary objects into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_dependency_versions(
    dependencies: list[tuple[str, str]] | None = None,
) -> dict[str, str | None]:
    deps = dependencies or _DEFAULT_DEPENDENCIES
    versions: dict[str, str | None] = {}
    for key, dist_name in deps:
        try:
            versions[key] = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = None
        except Exception:
            versions[key] = None
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
            "is_dirty": None,
        }

    root = Path(repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    status = _run_git(["status", "--porcelain"], cwd=root)
    is_dirty = None if status is None else bool(status.strip())
    return {
        "repo_root": str(root),
        "branch": branch,
        "commit": commit,
        "is_dirty": is_dirty,
    }


def _collect_artifacts(output_dir: Path, metadata_filename: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == metadata_filename:
            continue
        rel = path.relative_to(output_dir)
        try:
            digest = _sha256_file(path)
        except Exception:
            digest = None
        artifacts.append(
            {
                "path": str(rel),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return artifacts


def _build_input_fingerprints(input_payloads: dict[str, Any] | None) -> dict[str, Any]:
    if not input_payloads:
        return {}
    fingerprints: dict[str, Any] = {}
    for key, payload in input_payloads.items():
        try:
            normalized = _to_jsonable(payload)
            serialized = _stable_json_dumps(normalized)
            count = len(normalized) if hasattr(normalized, "__len__") else None
            fingerprints[key] = {
                "sha256": _sha256_text(serialized),
                "count": count,
            }
        except Exception:
            fingerprints[key] = {
                "sha256": None,
                "count": None,
            }
    return fingerprints


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
        "results": _to_jsonable(results or {}),
        "input_fingerprints": _build_input_fingerprints(input_payloads),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "user": os.getenv("USER", None),
        },
        "dependencies": _get_dependency_versions(),
        "git": _get_git_info(start_path=Path(__file__).resolve().parent),
    }
    if extras:
        metadata["extras"] = _to_jsonable(extras)

    artifacts = _collect_artifacts(output_path, metadata_filename=metadata_filename)
    metadata["artifacts"] = artifacts
    metadata["artifacts_fingerprint_sha256"] = _sha256_text(
        _stable_json_dumps(artifacts)
    )

    metadata_path = output_path / metadata_filename
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path

