"""Schema for multi-criteria rubric evaluation.

Defines the data structures for rubric criteria, complete rubrics,
and per-completion rubric scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Criterion:
    """A single scoring criterion in a rubric.

    Attributes:
        name: Short identifier (e.g. "fluency", "usefulness").
        description: Human-readable description shown to the judge.
        scale_min: Minimum score (inclusive).
        scale_max: Maximum score (inclusive).
        weight: Optional prior weight for aggregation (not used in BT fitting).
        score_references: Optional score anchors shown to the judge in the prompt.
            Mapping ``score -> description`` (e.g. ``{7: "...", 5: "...", 3: "...", 1: "..."}``).
    """

    name: str
    description: str
    scale_min: int = 1
    scale_max: int = 10
    weight: float = 1.0
    score_references: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scale_min >= self.scale_max:
            raise ValueError(
                f"Invalid rubric scale for '{self.name}': "
                f"{self.scale_min} > {self.scale_max}"
            )

        # JSON object keys arrive as strings; normalize to ints.
        normalized: dict[int, str] = {}
        for raw_score, text in self.score_references.items():
            score = int(raw_score)
            if not (self.scale_min <= score <= self.scale_max):
                raise ValueError(
                    f"Score reference {score} for '{self.name}' is outside "
                    f"the configured scale [{self.scale_min}, {self.scale_max}]"
                )
            normalized[score] = str(text).strip()
        self.score_references = normalized

    def prompt_block(self) -> str:
        """Render this criterion as a scoring instruction for the judge."""
        base = (
            f"**{self.name.title()}** ({self.scale_min}–{self.scale_max}): "
            f"{self.description}"
        )
        if not self.score_references:
            return base

        refs = "\n".join(
            f"   - {score}: {self.score_references[score]}"
            for score in sorted(self.score_references.keys(), reverse=True)
        )
        return f"{base}\n   Score references:\n{refs}"


RubricDimension = Criterion


@dataclass(init=False)
class Rubric:
    """A collection of scoring criteria forming a complete rubric.

    Attributes:
        name: Rubric identifier (e.g. "default", "my_custom").
        criteria: List of scoring criteria.
        description: Optional description of when to use this rubric.
    """

    name: str
    criteria: list[Criterion]
    description: str = ""

    def __init__(
        self,
        name: str,
        criteria: list[Criterion] | None = None,
        description: str = "",
        dimensions: list[Criterion] | None = None,
    ) -> None:
        """Create a rubric.

        ``criteria`` is the preferred argument name. ``dimensions`` is kept as a
        compatibility alias for older call sites.
        """
        if criteria is not None and dimensions is not None:
            raise TypeError(
                "Pass either 'criteria' (preferred) or legacy 'dimensions', not both."
            )

        resolved_criteria = criteria if criteria is not None else dimensions
        if resolved_criteria is None:
            raise TypeError(
                "Rubric requires 'criteria' (preferred) or legacy 'dimensions'."
            )

        self.name = name
        self.criteria = resolved_criteria
        self.description = description

    @property
    def dimensions(self) -> list[Criterion]:
        """Compatibility alias for ``criteria``."""
        return self.criteria

    @property
    def criterion_names(self) -> list[str]:
        """Names of all rubric criteria in order."""
        return [c.name for c in self.criteria]

    @property
    def dimension_names(self) -> list[str]:
        """Compatibility alias for ``criterion_names``."""
        return self.criterion_names

    @property
    def num_criteria(self) -> int:
        """Number of criteria."""
        return len(self.criteria)

    @property
    def k(self) -> int:
        """Compatibility alias for ``num_criteria``."""
        return self.num_criteria

    def prompt_block(self) -> str:
        """Render the full rubric as scoring instructions."""
        lines = ["Score the following completion on each dimension:\n"]
        for i, dim in enumerate(self.criteria, 1):
            lines.append(f"{i}. {dim.prompt_block()}")
        return "\n".join(lines)


@dataclass
class RubricScore:
    """Scores for a single (instruction, completion) pair across all criteria.

    Attributes:
        instruction_index: Index linking back to the instruction.
        model: Model that generated the completion.
        scores: Dict mapping criterion name → score.
        raw_judge_output: Raw text from the judge (for debugging).
    """

    instruction_index: int | str
    model: str
    scores: dict[str, float]
    raw_judge_output: str = ""

    def to_list(self, criterion_names: list[str]) -> list[float]:
        """Return scores as an ordered vector matching criterion_names."""
        return [self.scores.get(name, 0.0) for name in criterion_names]


@dataclass
class PairwiseRubricResult:
    """Result of a pairwise rubric comparison: both A and B scored in one call.

    The judge sees both completions side-by-side, scores each on every rubric
    criterion, and gives an overall preference.

    Attributes:
        instruction_index: Index linking back to the instruction.
        scores_A: Dict mapping criterion name → score for completion A.
        scores_B: Dict mapping criterion name → score for completion B.
        preference: Overall preference from the judge: 0.0 = A wins,
            0.5 = tie, 1.0 = B wins.
        raw_judge_output: Raw text from the judge (for debugging).
        raw_judge_output_swapped: Raw output from B/A order (if debiasing).
    """

    instruction_index: int | str
    scores_A: dict[str, float]
    scores_B: dict[str, float]
    preference: float  # 0.0 = A wins, 0.5 = tie, 1.0 = B wins
    raw_judge_output: str = ""
    raw_judge_output_swapped: str | None = None

    def as_rubric_scores(
        self, model_A: str, model_B: str,
    ) -> tuple[RubricScore, RubricScore]:
        """Convert to two RubricScore objects (one per model)."""
        return (
            RubricScore(
                instruction_index=self.instruction_index,
                model=model_A,
                scores=self.scores_A,
                raw_judge_output=self.raw_judge_output,
            ),
            RubricScore(
                instruction_index=self.instruction_index,
                model=model_B,
                scores=self.scores_B,
                raw_judge_output=self.raw_judge_output,
            ),
        )
