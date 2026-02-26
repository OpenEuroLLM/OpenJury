"""Score completions on rubric criteria using an LLM judge.

Supports two modes:

- **Pairwise** (default): Judge sees both A and B side-by-side, scores each
  on rubric criteria, then gives an overall preference. N calls (2N with
  swap-debiasing). Best for ranking.
- **Sample-wise**: Judge scores each completion independently (2N calls).
  Preferences are derived from the weighted average of rubric criterion
  scores — no extra judge call. Optionally uses a reference answer as an
  anchor. Best for absolute quality assessment.

Example (pairwise)::

    from openjury.rubrics.defaults import get_rubric
    from openjury.rubrics.scorer import RubricScorer
    from openjury.utils import make_model

    judge = make_model("VLLM/Qwen/Qwen3-32B")
    scorer = RubricScorer(judge_model=judge, rubric=get_rubric("default"))

    results = scorer.score_pairwise(
        instructions=["Write a haiku about AI"],
        completions_A=["Silicon dreams hum / ..."],
        completions_B=["AI is good / ..."],
    )

Example (sample-wise with reference)::

    sample_scorer = RubricScorer(
        judge_model=judge,
        rubric=get_rubric("default"),
        mode="samplewise",
    )

    scores = sample_scorer.score(
        instructions=["What is 2+2?"],
        completions=["4"],
        model_name="gpt-4o",
        reference_answers=["The answer is 4."],
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from openjury.prompts import load_prompt
from openjury.rubrics.schema import Rubric, RubricScore, PairwiseRubricResult
from openjury.utils import do_inference
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  Scorer
# ═════════════════════════════════════════════════════════════════════

_EXAMPLE_SCORE_SEED = (7, 8, 6, 9, 7, 8, 5, 10, 6, 9)


def _build_example_scores(
    rubric: Rubric,
    decrement: int = 0,
) -> dict[str, int]:
    """Build deterministic example scores using the rubric's criterion names."""
    scores: dict[str, int] = {}
    for i, dim in enumerate(rubric.criteria):
        base = _EXAMPLE_SCORE_SEED[i % len(_EXAMPLE_SCORE_SEED)] - decrement
        scores[dim.name] = min(dim.scale_max, max(dim.scale_min, base))
    return scores


def _build_example_json_strings(rubric: Rubric) -> tuple[str, str]:
    """Return example JSON strings for samplewise and pairwise prompt formatting."""
    example_a = _build_example_scores(rubric)
    example_b = _build_example_scores(rubric, decrement=1)
    example_json = json.dumps(example_a)
    example_pairwise_json = json.dumps(
        {"scores_A": example_a, "scores_B": example_b, "preference": "A"},
        indent=2,
    )
    return example_json, example_pairwise_json


class RubricScorer:
    """Scores completions along rubric criteria using an LLM judge.

    Supports pairwise (A vs B) and sample-wise (independent) scoring.

    Args:
        judge_model: A LangChain-compatible chat model or wrapper accepted by
            ``openjury.utils.do_inference``.
        rubric: The rubric to evaluate against.
        provide_explanation: Whether to ask the judge for explanations.
        mode: Scoring mode. ``"pairwise"`` (default) or ``"samplewise"``.
    """

    def __init__(
        self,
        judge_model: Any,
        rubric: Rubric,
        provide_explanation: bool = False,
        mode: Literal["pairwise", "samplewise"] = "pairwise",
    ):
        if mode not in {"pairwise", "samplewise"}:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'pairwise' or 'samplewise'.")
        self.judge_model = judge_model
        self.rubric = rubric
        self.provide_explanation = provide_explanation
        self.mode = mode

        explanation_block = (
            "Before providing scores, briefly explain your reasoning for each dimension."
            if provide_explanation
            else "Provide ONLY the JSON output, no explanation needed."
        )

        example_json, example_pairwise_json = _build_example_json_strings(rubric)
        if self.mode == "pairwise":
            self._init_pairwise_prompts(
                explanation_block=explanation_block,
                example_pairwise_json=example_pairwise_json,
            )
        else:
            self._init_samplewise_prompts(
                explanation_block=explanation_block,
                example_json=example_json,
            )

    def _init_pairwise_prompts(
        self,
        explanation_block: str,
        example_pairwise_json: str,
    ) -> None:
        """Load and format pairwise prompt templates only."""
        pairwise_system_template = load_prompt("rubric_pairwise_system")
        self._pairwise_user_template = load_prompt("rubric_pairwise_user")
        self._pairwise_system = pairwise_system_template.format(
            rubric_block=self.rubric.prompt_block(),
            explanation_block=explanation_block,
            example_json_pairwise=example_pairwise_json,
        )

    def _init_samplewise_prompts(
        self,
        explanation_block: str,
        example_json: str,
    ) -> None:
        """Load and format samplewise prompt templates only."""
        samplewise_system_template = load_prompt("rubric_samplewise_system")
        self._samplewise_user_template = load_prompt("rubric_samplewise_user")
        self._samplewise_system = samplewise_system_template.format(
            rubric_block=self.rubric.prompt_block(),
            reference_block="",  # Filled dynamically when reference is provided
            explanation_block=explanation_block,
            example_json=example_json,
        )
        self._samplewise_system_with_ref = samplewise_system_template.format(
            rubric_block=self.rubric.prompt_block(),
            reference_block=(
                "You are also given a REFERENCE ANSWER. Use it as an anchor to "
                "calibrate your scores. A perfect completion should match or exceed "
                "the reference answer in quality."
            ),
            explanation_block=explanation_block,
            example_json=example_json,
        )

    @property
    def system_prompt(self) -> dict[str, str]:
        """Return the formatted system prompts for debugging/reproducibility.

        Returns:
            Mode-specific prompt dict:
            - pairwise mode: ``{"pairwise": ...}``
            - samplewise mode: ``{"samplewise": ..., "samplewise_with_ref": ...}``
        """
        if self.mode == "pairwise":
            return {"pairwise": self._pairwise_system}
        return {
            "samplewise": self._samplewise_system,
            "samplewise_with_ref": self._samplewise_system_with_ref,
        }

    # ─────────────────────────────────────────────────────────────
    #  Key normalisation
    # ─────────────────────────────────────────────────────────────

    def _normalize_score_keys(
        self, scores: dict[str, float],
    ) -> dict[str, float]:
        """Map parsed score keys back to canonical rubric criterion names.

        The judge prompt renders criteria via ``prompt_block()`` which
        applies ``.title()`` casing (e.g. ``Instruction_Adherence``).  The
        model's JSON output therefore typically uses title-cased keys,
        but all downstream code (preference derivation, weighted
        averaging, DataFrame columns) expects the canonical lowercase
        names (``instruction_adherence``). This method performs a
        case-insensitive mapping to reconcile the two.
        """
        lookup = {d.lower(): d for d in self.rubric.criterion_names}
        return {
            lookup.get(k.lower(), k): v
            for k, v in scores.items()
        }

    # ─────────────────────────────────────────────────────────────
    #  Pairwise scoring (DEFAULT — recommended for ranking)
    # ─────────────────────────────────────────────────────────────

    def score_pairwise(
        self,
        instructions: list[str],
        completions_A: list[str],
        completions_B: list[str],
        swap_to_debias: bool = True,
        use_tqdm: bool = False,
        force_async: bool = False,
    ) -> list[PairwiseRubricResult]:
        """Compare A vs B on rubric criteria in a single judge call.

        The judge sees both completions, scores each on every criterion,
        and states an overall preference. If ``swap_to_debias=True``, runs
        each pair twice (A/B then B/A) and averages to cancel position bias.

        Args:
            instructions: List of instructions/prompts.
            completions_A: Completions from model A.
            completions_B: Completions from model B.
            swap_to_debias: Run A/B + B/A and average (recommended).
            use_tqdm: Show progress bar.
            force_async: Ignored on this branch; kept for API compatibility.

        Returns:
            List of PairwiseRubricResult, one per instruction.
        """
        if self.mode != "pairwise":
            raise RuntimeError(
                "RubricScorer is configured for samplewise mode. "
                "Instantiate with mode='pairwise' to use score_pairwise()."
            )
        assert len(instructions) == len(completions_A) == len(completions_B)
        n = len(instructions)

        # Build A/B prompts
        prompts_ab = self._build_pairwise_prompts(
            instructions, completions_A, completions_B,
        )

        logger.info(
            "Pairwise rubric scoring: %d pairs on %d criteria (A/B order)",
            n, self.rubric.num_criteria,
        )
        raw_ab = do_inference(
            chat_model=self.judge_model,
            inputs=prompts_ab,
            use_tqdm=use_tqdm,
        )

        # Optionally run B/A for debiasing
        raw_ba = None
        if swap_to_debias:
            prompts_ba = self._build_pairwise_prompts(
                instructions=instructions,
                completions_A=completions_B,  # SWAPPED
                completions_B=completions_A,  # SWAPPED
            )
            logger.info(
                "Pairwise rubric scoring: %d pairs on %d criteria (B/A debiasing)",
                n, self.rubric.num_criteria,
            )
            raw_ba = do_inference(
                chat_model=self.judge_model,
                inputs=prompts_ba,
                use_tqdm=use_tqdm,
            )

        # Parse and merge results
        results: list[PairwiseRubricResult] = []
        for i in range(n):
            text_ab = raw_ab[i] if isinstance(raw_ab[i], str) else raw_ab[i].content
            parsed_ab = self._parse_pairwise(text_ab)

            if swap_to_debias and raw_ba is not None:
                text_ba = raw_ba[i] if isinstance(raw_ba[i], str) else raw_ba[i].content
                parsed_ba = self._parse_pairwise(text_ba)
                merged = self._merge_swapped(parsed_ab, parsed_ba)
                merged_raw_swapped = text_ba
            else:
                merged = parsed_ab
                merged_raw_swapped = None

            results.append(PairwiseRubricResult(
                instruction_index=i,
                scores_A=merged["scores_A"],
                scores_B=merged["scores_B"],
                preference=merged["preference"],
                raw_judge_output=text_ab,
                raw_judge_output_swapped=merged_raw_swapped,
            ))

        valid = sum(1 for r in results if r.preference != 0.5)
        logger.info(
            "Pairwise rubric: %d/%d valid preferences%s",
            valid, n,
            " (position-swap debiased)" if swap_to_debias else "",
        )

        return results

    def _build_pairwise_prompts(
        self,
        instructions: list[str],
        completions_A: list[str],
        completions_B: list[str],
    ) -> list[list[tuple[str, str]]]:
        """Build chat prompts for pairwise comparison."""
        prompts = []
        for instr, comp_a, comp_b in zip(instructions, completions_A, completions_B):
            user_msg = self._pairwise_user_template.format(
                instruction=instr,
                completion_A=comp_a,
                completion_B=comp_b,
            )
            prompts.append([
                ("system", self._pairwise_system),
                ("user", user_msg),
            ])
        return prompts

    def _parse_pairwise(self, raw_output: str) -> dict:
        """Parse pairwise judge output into scores_A, scores_B, preference.

        Returns:
            Dict with keys: scores_A, scores_B (dicts), preference (float).
        """
        nan_scores = {dim: float("nan") for dim in self.rubric.criterion_names}
        default = {"scores_A": nan_scores.copy(), "scores_B": nan_scores.copy(), "preference": 0.5}

        # Try ```json ... ``` blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._extract_pairwise_from_dict(data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: find any large JSON object
        json_match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._extract_pairwise_from_dict(data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: try regex for individual score patterns
        scores_a, scores_b = {}, {}
        for dim in self.rubric.criterion_names:
            # Look for patterns like scores_A.fluency: 4 or "A": {"fluency": 4}
            match_a = re.search(
                rf'(?:scores?_?A|completion_?A).*?{re.escape(dim)}.*?(\d+(?:\.\d+)?)',
                raw_output, re.IGNORECASE | re.DOTALL,
            )
            match_b = re.search(
                rf'(?:scores?_?B|completion_?B).*?{re.escape(dim)}.*?(\d+(?:\.\d+)?)',
                raw_output, re.IGNORECASE | re.DOTALL,
            )
            if match_a:
                scores_a[dim] = float(match_a.group(1))
            if match_b:
                scores_b[dim] = float(match_b.group(1))

        pref_match = re.search(r'preference["\s:]*["\']?(A|B|tie)', raw_output, re.IGNORECASE)
        pref = 0.5
        if pref_match:
            p = pref_match.group(1).upper()
            pref = 0.0 if p == "A" else 1.0 if p == "B" else 0.5

        if scores_a or scores_b:
            for dim in self.rubric.criterion_names:
                scores_a.setdefault(dim, float("nan"))
                scores_b.setdefault(dim, float("nan"))
            return {"scores_A": scores_a, "scores_B": scores_b, "preference": pref}

        logger.warning("Could not parse pairwise rubric output: %s", raw_output[:200])
        return default

    def _extract_pairwise_from_dict(self, data: dict) -> dict:
        """Extract pairwise scores from a parsed JSON dict."""
        scores_a = data.get("scores_A", data.get("scores_a", {}))
        scores_b = data.get("scores_B", data.get("scores_b", {}))
        pref_raw = data.get("preference", "tie")

        if isinstance(pref_raw, str):
            pref_raw = pref_raw.strip().upper()
            pref = 0.0 if pref_raw == "A" else 1.0 if pref_raw == "B" else 0.5
        elif isinstance(pref_raw, (int, float)):
            pref = float(pref_raw)
        else:
            pref = 0.5

        scores_a = self._normalize_score_keys({k: float(v) for k, v in scores_a.items()})
        scores_b = self._normalize_score_keys({k: float(v) for k, v in scores_b.items()})

        return {"scores_A": scores_a, "scores_B": scores_b, "preference": pref}

    def _merge_swapped(self, ab: dict, ba: dict) -> dict:
        """Merge A/B and B/A (swapped) results to cancel position bias.

        In the B/A run, positions are swapped: what the judge calls "A" is
        actually B, and vice versa. So we flip them back before averaging.
        """
        merged_A, merged_B = {}, {}
        for dim in self.rubric.criterion_names:
            # ab: scores_A = actual A, scores_B = actual B
            # ba: scores_A = actual B (swapped), scores_B = actual A (swapped)
            a_from_ab = ab["scores_A"].get(dim, float("nan"))
            a_from_ba = ba["scores_B"].get(dim, float("nan"))  # flip back
            b_from_ab = ab["scores_B"].get(dim, float("nan"))
            b_from_ba = ba["scores_A"].get(dim, float("nan"))  # flip back

            merged_A[dim] = float(np.nanmean([a_from_ab, a_from_ba]))
            merged_B[dim] = float(np.nanmean([b_from_ab, b_from_ba]))

        # Merge preference: ab.preference is P(B wins in A/B order)
        # ba.preference is P(A-actual wins in B/A order) = P(B wins) in swapped = 1 - P(A wins)
        pref_ab = ab["preference"]
        pref_ba = 1.0 - ba["preference"]  # Flip back
        merged_pref = (pref_ab + pref_ba) / 2.0

        return {"scores_A": merged_A, "scores_B": merged_B, "preference": merged_pref}

    # ─────────────────────────────────────────────────────────────
    #  Sample-wise scoring (independent, optional reference anchor)
    # ─────────────────────────────────────────────────────────────

    def _build_prompts(
        self,
        instructions: list[str],
        completions: list[str],
        reference_answers: list[str] | None = None,
    ) -> list[list[tuple[str, str]]]:
        """Build chat prompts for sample-wise scoring."""
        has_ref = reference_answers is not None
        system = self._samplewise_system_with_ref if has_ref else self._samplewise_system

        prompts = []
        for i, (instruction, completion) in enumerate(zip(instructions, completions)):
            ref_section = ""
            if has_ref and reference_answers[i]:
                ref_section = f"## Reference Answer\n{reference_answers[i]}"

            user_msg = self._samplewise_user_template.format(
                instruction=instruction,
                reference_section=ref_section,
                completion=completion,
            )
            prompts.append([
                ("system", system),
                ("user", user_msg),
            ])
        return prompts

    def  _parse_scores(self, raw_output: str) -> dict[str, float]:
        """Extract criterion scores from judge output.

        Tries to find JSON in ```json``` blocks first, then falls back to
        finding any JSON object in the text, and finally to key: value patterns.

        Args:
            raw_output: Raw text output from the judge.

        Returns:
            Dict mapping criterion name → score. NaN for unparseable scores.
        """
        # Try ```json ... ``` blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(1))
                return self._normalize_score_keys(
                    {k: float(v) for k, v in scores.items()}
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: find any JSON object
        json_match = re.search(r"\{[^{}]*\}", raw_output)
        if json_match:
            try:
                scores = json.loads(json_match.group(0))
                return self._normalize_score_keys(
                    {k: float(v) for k, v in scores.items()}
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: try to parse key: value patterns
        scores = {}
        for dim in self.rubric.criterion_names:
            pattern = rf'["\']?{re.escape(dim)}["\']?\s*[:=]\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                scores[dim] = float(match.group(1))

        if scores:
            return scores

        logger.warning(
            "Could not parse rubric scores from judge output: %s",
            raw_output[:200],
        )
        return {dim: float("nan") for dim in self.rubric.criterion_names}

    def score(
        self,
        instructions: list[str],
        completions: list[str],
        model_name: str,
        use_tqdm: bool = False,
        force_async: bool = False,
        reference_answers: list[str] | None = None,
    ) -> list[RubricScore]:
        """Score completions independently (sample-wise) on the rubric.

        Each completion is evaluated in isolation. If ``reference_answers``
        is provided, the judge uses them as quality anchors for calibration.

        Args:
            instructions: List of instructions/prompts.
            completions: List of model completions.
            model_name: Name of the model that generated the completions.
            use_tqdm: Whether to show progress bar.
            force_async: Ignored on this branch; kept for API compatibility.
            reference_answers: Optional list of reference/ground-truth answers
                to anchor scoring. Can come from another LLM or human labels.

        Returns:
            List of RubricScore objects, one per (instruction, completion) pair.
        """
        if self.mode != "samplewise":
            raise RuntimeError(
                "RubricScorer is configured for pairwise mode. "
                "Instantiate with mode='samplewise' to use score()."
            )
        assert len(instructions) == len(completions), (
            f"instructions ({len(instructions)}) and completions ({len(completions)}) "
            "must have the same length"
        )
        if reference_answers is not None:
            assert len(reference_answers) == len(instructions), (
                f"reference_answers ({len(reference_answers)}) must match "
                f"instructions ({len(instructions)})"
            )

        prompts = self._build_prompts(instructions, completions, reference_answers)

        mode_label = "sample-wise (with reference)" if reference_answers else "sample-wise"
        logger.info(
            "Scoring %d completions from [model]%s[/model] on %d rubric criteria (%s)",
            len(prompts), model_name, self.rubric.num_criteria, mode_label,
        )

        raw_outputs = do_inference(
            chat_model=self.judge_model,
            inputs=prompts,
            use_tqdm=use_tqdm,
        )

        results = []
        for i, raw in enumerate(raw_outputs):
            raw_text = raw if isinstance(raw, str) else raw.content
            scores = self._parse_scores(raw_text)
            results.append(
                RubricScore(
                    instruction_index=i,
                    model=model_name,
                    scores=scores,
                    raw_judge_output=raw_text,
                )
            )

        valid_scores = [
            r
            for r in results
            if not any(v != v for v in r.scores.values())  # NaN check
        ]
        logger.info(
            "Successfully parsed %d/%d rubric scores",
            len(valid_scores),
            len(results),
        )

        return results

    # ─────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────

    def score_to_dataframe(
        self,
        rubric_scores: list[RubricScore],
    ) -> pd.DataFrame:
        """Convert rubric scores to a DataFrame.

        Columns: instruction_index, model, criterion_1, criterion_2, ..., raw_judge_output.
        """
        rows = []
        for rs in rubric_scores:
            row = {
                "instruction_index": rs.instruction_index,
                "model": rs.model,
                **rs.scores,
                "raw_judge_output": rs.raw_judge_output,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def pairwise_to_dataframes(
        self,
        results: list[PairwiseRubricResult],
        model_A_name: str,
        model_B_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Convert pairwise results to DataFrames + preferences.

        Returns:
            (df_scores_A, df_scores_B, preferences) where preferences is a
            Series of floats (0.0 = A wins, 0.5 = tie, 1.0 = B wins).
        """
        rows_A, rows_B = [], []
        prefs = []
        for r in results:
            rows_A.append({
                "instruction_index": r.instruction_index,
                "model": model_A_name,
                **r.scores_A,
            })
            rows_B.append({
                "instruction_index": r.instruction_index,
                "model": model_B_name,
                **r.scores_B,
            })
            prefs.append(r.preference)

        return pd.DataFrame(rows_A), pd.DataFrame(rows_B), pd.Series(prefs)

    def samplewise_to_dataframes(
        self,
        scores_A: list[RubricScore],
        scores_B: list[RubricScore],
        model_A_name: str,
        model_B_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Convert samplewise scores to DataFrames and derive preferences.

        Preferences are computed from the **weighted average** of rubric
        criterion scores — no extra judge call required. For each
        instruction, if weighted-avg(A) > weighted-avg(B) → 0.0 (A wins),
        < → 1.0 (B wins), equal → 0.5 (tie).  Weights come from
        ``Criterion.weight`` (all 1.0 by default, i.e. plain mean).

        Returns:
            (df_scores_A, df_scores_B, preferences)
        """
        df_A = self.score_to_dataframe(scores_A)
        df_B = self.score_to_dataframe(scores_B)

        weights = {d.name: d.weight for d in self.rubric.criteria}

        prefs: list[float] = []
        for sa, sb in zip(scores_A, scores_B):
            avg_a = self._weighted_rubric_avg(sa.scores, weights)
            avg_b = self._weighted_rubric_avg(sb.scores, weights)

            if np.isnan(avg_a) or np.isnan(avg_b):
                prefs.append(0.5)          # can't decide — treat as tie
            elif avg_a > avg_b:
                prefs.append(0.0)          # A wins
            elif avg_b > avg_a:
                prefs.append(1.0)          # B wins
            else:
                prefs.append(0.5)          # tie

        prefs_series = pd.Series(prefs)

        n_a = (prefs_series < 0.5).sum()
        n_b = (prefs_series > 0.5).sum()
        n_t = (prefs_series == 0.5).sum()
        logger.info(
            "Samplewise preferences (from rubric weighted avg): "
            "A wins=%d, B wins=%d, ties=%d (of %d)",
            n_a, n_b, n_t, len(prefs),
        )

        return df_A, df_B, prefs_series

    @staticmethod
    def _weighted_rubric_avg(
        scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Weighted average of rubric scores, skipping NaN criteria."""
        total_w = 0.0
        total_s = 0.0
        for dim, w in weights.items():
            s = scores.get(dim, float("nan"))
            if s == s:  # not NaN
                total_w += w
                total_s += w * s
        return total_s / total_w if total_w > 0 else float("nan")
