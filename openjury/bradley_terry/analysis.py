"""Analysis utilities for fitted Bradley-Terry models.

Supports slicing by annotator type, language, topic, or any other
metadata column to compare which quality dimensions drive preferences
for different populations.

Example::

    analysis = BTAnalysis(model=bt, scores_A=df_A, scores_B=df_B,
                          preferences=prefs, metadata=meta_df)
    table = analysis.weight_comparison_table("annotator_type")
    print(BTAnalysis.summary_report(table))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from openjury._logging import logger
from openjury.bradley_terry.model import FeatureBradleyTerry


@dataclass
class BTAnalysis:
    """Analyze Bradley-Terry weights across different data slices.

    This class helps answer questions like:
        - Do LLM judges and human annotators weight criteria differently?
        - Does fluency matter more in lower-resource languages?
        - Does adherence matter more for coding tasks?

    Attributes:
        model: A fitted (or unfitted) FeatureBradleyTerry providing dimension_names
               and regularization settings.
        scores_A: DataFrame of rubric scores for model A completions.
        scores_B: DataFrame of rubric scores for model B completions.
        preferences: Series of pairwise preferences (0.0 = prefer A, 0.5 = tie, 1.0 = prefer B).
        metadata: DataFrame with grouping columns (e.g. ``annotator_type``,
                  ``language``, ``topic``).  Must be aligned with the other inputs.
    """

    model: FeatureBradleyTerry
    scores_A: pd.DataFrame
    scores_B: pd.DataFrame
    preferences: pd.Series
    metadata: pd.DataFrame

    def fit_by_group(
        self,
        group_column: str,
        min_samples: int = 10,
        **fit_kwargs,
    ) -> dict[str, FeatureBradleyTerry]:
        """Fit separate BT models for each group in a metadata column.

        Args:
            group_column: Column in ``self.metadata`` to group by
                          (e.g. ``"annotator_type"``, ``"language"``, ``"topic"``).
            min_samples: Minimum number of samples required per group.
            **fit_kwargs: Passed to ``FeatureBradleyTerry.fit()``.

        Returns:
            Dict mapping group name → fitted BT model.
        """
        assert group_column in self.metadata.columns, (
            f"Column '{group_column}' not in metadata. "
            f"Available: {list(self.metadata.columns)}"
        )

        groups = self.metadata[group_column].unique()
        results: dict[str, FeatureBradleyTerry] = {}

        for group in sorted(groups):
            mask = self.metadata[group_column] == group
            n = int(mask.sum())

            if n < min_samples:
                logger.warning(
                    "Skipping group %s=%s — only %d samples (need %d)",
                    group_column,
                    group,
                    n,
                    min_samples,
                )
                continue

            logger.info(
                "Fitting BT model for %s=%s (%d samples)",
                group_column,
                group,
                n,
            )

            bt = FeatureBradleyTerry(
                dimension_names=self.model.dimension_names,
                regularization=self.model.regularization,
            )
            bt.fit(
                scores_A=self.scores_A[mask].reset_index(drop=True),
                scores_B=self.scores_B[mask].reset_index(drop=True),
                preferences=self.preferences[mask].reset_index(drop=True),
                **fit_kwargs,
            )
            results[str(group)] = bt

        return results

    def weight_comparison_table(
        self,
        group_column: str,
        **fit_kwargs,
    ) -> pd.DataFrame:
        """Create a comparison table of BT weights across groups.

        Returns a DataFrame like::

                            fluency  usefulness  clarity  style  adherence
            LLM-judge         0.32      0.85      0.41    0.12     0.55
            human-lmsys       0.45      0.72      0.38    0.31     0.48
            human-comparia    0.61      0.55      0.29    0.42     0.39

        Args:
            group_column: Metadata column to group by.
            **fit_kwargs: Passed to ``FeatureBradleyTerry.fit()``.

        Returns:
            DataFrame with groups as rows and dimensions as columns.
        """
        fitted_models = self.fit_by_group(group_column, **fit_kwargs)

        rows = {}
        for group_name, bt in fitted_models.items():
            rows[group_name] = bt.weight_dict()

        df = pd.DataFrame(rows).T
        df.index.name = group_column
        return df

    def compute_accuracy(
        self,
        scores_A: pd.DataFrame | None = None,
        scores_B: pd.DataFrame | None = None,
        preferences: pd.Series | None = None,
    ) -> dict[str, float]:
        """Compute prediction accuracy metrics for the fitted model.

        Args:
            scores_A: Optional override; defaults to ``self.scores_A``.
            scores_B: Optional override; defaults to ``self.scores_B``.
            preferences: Optional override; defaults to ``self.preferences``.

        Returns:
            Dict with ``accuracy``, ``log_loss``, and ``n_samples``.
        """
        if scores_A is None:
            scores_A = self.scores_A
        if scores_B is None:
            scores_B = self.scores_B
        if preferences is None:
            preferences = self.preferences

        proba = self.model.predict_proba(scores_A, scores_B)
        preds = (proba >= 0.5).astype(int)
        # predict_proba returns P(A≻B): high = A wins
        # prefs convention: 0.0 = A wins, 1.0 = B wins
        labels = (preferences.values < 0.5).astype(int)  # 1 = A wins

        accuracy = float((preds == labels).mean())

        # Log-loss
        eps = 1e-12
        ll = float(
            -np.mean(
                labels * np.log(proba + eps)
                + (1 - labels) * np.log(1 - proba + eps)
            )
        )

        return {
            "accuracy": accuracy,
            "log_loss": ll,
            "n_samples": len(labels),
        }

    @staticmethod
    def summary_report(
        weight_table: pd.DataFrame,
        title: str = "Bradley-Terry Weight Analysis",
    ) -> str:
        """Generate a text summary of weight differences across groups.

        Args:
            weight_table: Output of ``weight_comparison_table()``.
            title: Title for the report header.

        Returns:
            Multi-line string with formatted summary.
        """
        lines = [f"\n{'=' * 60}", f"  {title}", f"{'=' * 60}\n"]
        lines.append(weight_table.to_string(float_format="%.4f"))
        lines.append("")

        # Most important dimension per group
        for group in weight_table.index:
            top_dim = weight_table.loc[group].idxmax()
            top_val = weight_table.loc[group].max()
            lines.append(
                f"  {group}: most predictive = {top_dim} (w={top_val:.4f})"
            )

        # Dimension with largest variance across groups
        if len(weight_table) > 1:
            variances = weight_table.var()
            most_variable = variances.idxmax()
            lines.append(
                f"\n  Most variable across groups: {most_variable} "
                f"(var={variances[most_variable]:.4f})"
            )

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)
