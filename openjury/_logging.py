"""Centralized logging for OpenJury using Rich.

Provides colored, structured console output with proper log levels.
New modules should prefer ``from openjury._logging import logger`` instead of ``print()``.

Environment variable ``OPENJURY_LOG_LEVEL`` controls verbosity:
    - DEBUG, INFO (default), WARNING, ERROR
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# ── Custom theme ──────────────────────────────────────────────────────────────
_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "model": "bold magenta",
        "dataset": "bold blue",
        "path": "dim",
        "number": "bold yellow",
    }
)

console = Console(theme=_theme)


def _setup_logger() -> logging.Logger:
    """Create and configure the openjury logger with Rich handler."""
    log_level = os.environ.get("OPENJURY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    _logger = logging.getLogger("openjury")
    _logger.setLevel(level)
    _logger.propagate = False

    # Avoid duplicate handlers when modules are reloaded.
    if not any(isinstance(h, RichHandler) for h in _logger.handlers):
        handler = RichHandler(
            console=console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        handler.setLevel(level)
        _logger.addHandler(handler)
    else:
        for handler in _logger.handlers:
            handler.setLevel(level)

    return _logger


logger = _setup_logger()


def print_results(results: dict) -> None:
    """Print battle results as a Rich panel with colored output."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="bold")
    table.add_column("value")

    table.add_row("Dataset", f"[dataset]{results['dataset']}[/dataset]")
    table.add_row("Model A", f"[model]{results['model_A']}[/model]")
    table.add_row("Model B", f"[model]{results['model_B']}[/model]")
    table.add_row("Judge", f"[model]{results['judge_model']}[/model]")
    table.add_row("", "")
    table.add_row("Total Battles", f"[number]{results['num_battles']}[/number]")
    table.add_row("Win Rate (A)", f"[number]{results['winrate']:.1%}[/number]")
    table.add_row("Wins", f"[success]{results['num_wins']}[/success]")
    table.add_row("Losses", f"[error]{results['num_losses']}[/error]")
    table.add_row("Ties", f"[number]{results['num_ties']}[/number]")

    panel = Panel(
        table,
        title="[bold]MODEL BATTLE RESULTS[/bold]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
