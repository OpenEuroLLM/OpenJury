"""Score completions on a criteria set using an LLM judge.

This scorer is intentionally sample-wise only:
- Each completion is scored independently against the same criteria.
- Preferences between model A and model B are derived later from weighted
  criterion averages (no direct head-to-head judge call).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from openjury.prompts import load_prompt
from openjury.criteria.schema import Criteria, CriteriaScore
from openjury.utils import do_inference

logger = logging.getLogger(__name__)


_EXAMPLE_SCORE_SEED = (7, 8, 6, 9, 7, 8, 5, 10, 6, 9)


def _build_example_scores(
    criteria: Criteria,
    decrement: int = 0,
) -> dict[str, int]:
    """Build deterministic example scores using criteria names."""
    scores: dict[str, int] = {}
    for i, criterion in enumerate(criteria.criteria):
        base = _EXAMPLE_SCORE_SEED[i % len(_EXAMPLE_SCORE_SEED)] - decrement
        scores[criterion.name] = min(criterion.scale_max, max(criterion.scale_min, base))
    return scores


def _build_example_json_strings(criteria: Criteria) -> str:
    """Return the sample-wise example JSON string used in prompts."""
    example = _build_example_scores(criteria)
    return json.dumps(example)


class CriteriaScorer:
    """Score completions along a criteria set using an LLM judge.

    Sample-wise only: each completion is evaluated independently.

    Args:
        judge_model: A LangChain-compatible chat model or wrapper accepted by
            ``openjury.utils.do_inference``.
        criteria: The criteria set to evaluate against.
        provide_explanation: Whether to ask the judge for explanations.
    """

    def __init__(
        self,
        judge_model: Any,
        criteria: Criteria,
        provide_explanation: bool = False,
    ):
        self.judge_model = judge_model
        self.criteria = criteria
        self.provide_explanation = provide_explanation

        explanation_block = (
            "Before providing scores, briefly explain your reasoning for each criterion."
            if provide_explanation
            else "Provide ONLY the JSON output, no explanation needed."
        )

        example_json = _build_example_json_strings(criteria)
        samplewise_system_template = load_prompt("criteria_samplewise_system")
        self._samplewise_user_template = load_prompt("criteria_samplewise_user")
        self._samplewise_system = samplewise_system_template.format(
            criteria_block=self.criteria.prompt_block(),
            reference_block="",  # Filled dynamically when reference is provided
            explanation_block=explanation_block,
            example_json=example_json,
        )
        self._samplewise_system_with_ref = samplewise_system_template.format(
            criteria_block=self.criteria.prompt_block(),
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
        """Return formatted sample-wise prompts for reproducibility/debugging."""
        return {
            "samplewise": self._samplewise_system,
            "samplewise_with_ref": self._samplewise_system_with_ref,
        }

    def _normalize_score_keys(self, scores: dict[str, float]) -> dict[str, float]:
        """Map parsed keys back to canonical criterion names (case-insensitive)."""
        lookup = {name.lower(): name for name in self.criteria.criterion_names}
        return {lookup.get(k.lower(), k): v for k, v in scores.items()}

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

    def _parse_scores(self, raw_output: str) -> dict[str, float]:
        """Extract criterion scores from judge output."""
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(1))
                return self._normalize_score_keys({k: float(v) for k, v in scores.items()})
            except (json.JSONDecodeError, ValueError):
                pass

        json_match = re.search(r"\{[^{}]*\}", raw_output)
        if json_match:
            try:
                scores = json.loads(json_match.group(0))
                return self._normalize_score_keys({k: float(v) for k, v in scores.items()})
            except (json.JSONDecodeError, ValueError):
                pass

        scores = {}
        for criterion_name in self.criteria.criterion_names:
            pattern = rf'["\']?{re.escape(criterion_name)}["\']?\s*[:=]\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                scores[criterion_name] = float(match.group(1))

        if scores:
            return scores

        logger.warning(
            "Could not parse criteria scores from judge output: %s",
            raw_output[:200],
        )
        return {criterion_name: float("nan") for criterion_name in self.criteria.criterion_names}

    def score(
        self,
        instructions: list[str],
        completions: list[str],
        model_name: str,
        use_tqdm: bool = False,
        force_async: bool = False,
        reference_answers: list[str] | None = None,
    ) -> list[CriteriaScore]:
        """Score completions independently (sample-wise) on the criteria."""
        if force_async:
            logger.debug("force_async=True is ignored in this branch.")

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
            "Scoring %d completions from %s on %d criteria (%s)",
            len(prompts),
            model_name,
            self.criteria.num_criteria,
            mode_label,
        )

        raw_outputs = do_inference(
            chat_model=self.judge_model,
            inputs=prompts,
            use_tqdm=use_tqdm,
        )

        results: list[CriteriaScore] = []
        for i, raw in enumerate(raw_outputs):
            raw_text = raw if isinstance(raw, str) else raw.content
            scores = self._parse_scores(raw_text)
            results.append(
                CriteriaScore(
                    instruction_index=i,
                    model=model_name,
                    scores=scores,
                    raw_judge_output=raw_text,
                )
            )

        valid_scores = [
            r
            for r in results
            if not any(v != v for v in r.scores.values())
        ]
        logger.info(
            "Successfully parsed %d/%d criteria scores",
            len(valid_scores),
            len(results),
        )

        return results

    def score_to_dataframe(self, criteria_scores: list[CriteriaScore]) -> pd.DataFrame:
        """Convert criteria scores to a DataFrame."""
        rows = []
        for rs in criteria_scores:
            rows.append(
                {
                    "instruction_index": rs.instruction_index,
                    "model": rs.model,
                    **rs.scores,
                    "raw_judge_output": rs.raw_judge_output,
                }
            )
        return pd.DataFrame(rows)

    def samplewise_to_dataframes(
        self,
        scores_A: list[CriteriaScore],
        scores_B: list[CriteriaScore],
        model_A_name: str,
        model_B_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Convert sample-wise scores to DataFrames and derived preferences."""
        df_A = self.score_to_dataframe(scores_A)
        df_B = self.score_to_dataframe(scores_B)

        weights = {criterion.name: criterion.weight for criterion in self.criteria.criteria}

        prefs: list[float] = []
        for sa, sb in zip(scores_A, scores_B):
            avg_a = self._weighted_criteria_avg(sa.scores, weights)
            avg_b = self._weighted_criteria_avg(sb.scores, weights)

            if np.isnan(avg_a) or np.isnan(avg_b):
                prefs.append(0.5)
            elif avg_a > avg_b:
                prefs.append(0.0)
            elif avg_b > avg_a:
                prefs.append(1.0)
            else:
                prefs.append(0.5)

        prefs_series = pd.Series(prefs)

        n_a = (prefs_series < 0.5).sum()
        n_b = (prefs_series > 0.5).sum()
        n_t = (prefs_series == 0.5).sum()
        logger.info(
            "Samplewise preferences (from criteria weighted avg): "
            "A wins=%d, B wins=%d, ties=%d (of %d)",
            n_a,
            n_b,
            n_t,
            len(prefs),
        )

        return df_A, df_B, prefs_series

    @staticmethod
    def _weighted_criteria_avg(
        scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Weighted average of criteria scores, skipping NaN criteria."""
        total_w = 0.0
        total_s = 0.0
        for criterion_name, weight in weights.items():
            score = scores.get(criterion_name, float("nan"))
            if score == score:
                total_w += weight
                total_s += weight * score
        return total_s / total_w if total_w > 0 else float("nan")
