"""Unified prompt-template loader for OpenJury.

All ``.txt`` files under ``openjury/prompts/`` are discoverable by stem name::

    from openjury.prompts import load_prompt

    system = load_prompt("criteria_pairwise_system")    # reads criteria_pairwise_system.txt
    user   = load_prompt("criteria_pairwise_user")      # reads criteria_pairwise_user.txt

Legacy templates from the pre-criteria pairwise scorer are also discoverable
here for backward compatibility and migration reference.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Load a prompt template by stem name (without ``.txt`` extension).

    Returns the raw text content — callers are responsible for calling
    ``.format(...)`` to fill in placeholders.

    Raises:
        FileNotFoundError: If no matching ``.txt`` file exists.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in _PROMPTS_DIR.glob("*.txt")))
        raise FileNotFoundError(
            f"Prompt template '{name}' not found in {_PROMPTS_DIR}. "
            f"Available: {available}"
        )
    return path.read_text(encoding="utf-8")


def list_prompts() -> list[str]:
    """Return the stem names of all available prompt templates."""
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.txt"))
