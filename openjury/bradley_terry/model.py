"""Bradley-Terry model augmented with interpretable rubric features.

Implements::

    p(A ≻ B | x) = σ(wᵀ [Φ(x, A) - Φ(x, B)])

where ``Φ(x, y)`` is a vector of rubric scores and ``w`` are learned weights.

This module requires the optional BT dependency extra:
    - ``uv sync --extra bt``
    - ``pip install -e '.[bt]'``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureBradleyTerry:
    """Bradley-Terry model with interpretable feature weights.

    Wraps an sklearn estimator (default: ``LogisticRegression``) that operates
    on standardized rubric-score differences ``Φ(A) - Φ(B)``.

    Preference convention:
        ``0.0 = A wins``, ``0.5 = tie``, ``1.0 = B wins``.

    Ties are removed before fitting using ``tie_epsilon``:
        a sample is treated as tie if ``abs(pref - 0.5) <= tie_epsilon``.
    """

    dimension_names: list[str]
    regularization: float = 0.01
    tie_epsilon: float = 0.05
    estimator: Any = None
    default_estimator_kwargs: dict[str, Any] = field(default_factory=dict)

    weights: np.ndarray | None = None
    intercept: float = 0.0
    fit_history: list[float] = field(default_factory=list)

    # Internal (not serialized)
    _scaler: StandardScaler = field(default_factory=StandardScaler, repr=False)
    _model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.tie_epsilon < 0.5):
            raise ValueError(
                f"tie_epsilon must be in [0.0, 0.5), got {self.tie_epsilon}"
            )
        forbidden = {"C", "fit_intercept"}
        overlap = forbidden.intersection(self.default_estimator_kwargs.keys())
        if overlap:
            raise ValueError(
                "default_estimator_kwargs cannot override managed parameters: "
                + ", ".join(sorted(overlap))
            )

    @property
    def k(self) -> int:
        """Number of feature dimensions."""
        return len(self.dimension_names)

    def _prepare_data(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
        preferences: pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Prepare feature matrices and labels.

        Args:
            scores_A: Rubric scores for model A.
            scores_B: Rubric scores for model B.
            preferences: ``0.0 = A wins``, ``1.0 = B wins``, ``0.5 = tie``.
                If ``None``, returns only feature matrices (for prediction).

        Returns:
            ``(phi_A, phi_B, y)`` where ``y`` is binary labels
            (``1 = A wins``) with ties removed, or ``None``.
        """
        phi_A = scores_A[self.dimension_names].values.astype(float)
        phi_B = scores_B[self.dimension_names].values.astype(float)

        if preferences is None:
            return phi_A, phi_B, None

        y_raw = preferences.values.astype(float)

        valid = ~(
            np.isnan(phi_A).any(axis=1)
            | np.isnan(phi_B).any(axis=1)
            | np.isnan(y_raw)
        )
        n_dropped_nan = int((~valid).sum())
        if n_dropped_nan > 0:
            logger.warning("Dropped %d/%d rows with NaN values", n_dropped_nan, len(valid))
        phi_A = phi_A[valid]
        phi_B = phi_B[valid]
        y_raw = y_raw[valid]

        # Treat values near 0.5 as ties and drop them before fitting.
        not_tie = np.abs(y_raw - 0.5) > self.tie_epsilon
        n_ties = int((~not_tie).sum())
        if n_ties > 0:
            logger.info(
                "Removed %d ties from %d samples using tie_epsilon=%.3f",
                n_ties,
                len(y_raw),
                self.tie_epsilon,
            )
        phi_A = phi_A[not_tie]
        phi_B = phi_B[not_tie]
        y_raw = y_raw[not_tie]

        # scorer convention 0.0 = A wins -> class 1 = A wins
        y = (y_raw < 0.5).astype(float)
        return phi_A, phi_B, y

    def fit(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
        preferences: pd.Series,
        fit_intercept: bool = True,
        verbose: bool = True,
        **estimator_kwargs,
    ) -> FeatureBradleyTerry:
        """Fit the model on rubric score differences.

        Args:
            scores_A: Rubric scores for model A.
            scores_B: Rubric scores for model B.
            preferences: ``0.0 = A wins``, ``1.0 = B wins``, ``0.5 = tie``.
            fit_intercept: Whether to fit a bias term.
            verbose: Whether to log fit details.
            **estimator_kwargs: Forwarded to the default ``LogisticRegression``
                constructor when ``self.estimator`` is not provided. Useful for
                overriding settings such as ``max_iter`` or ``solver``.

        Returns:
            ``self``.
        """
        forbidden = {"C", "fit_intercept"}
        overlap = forbidden.intersection(estimator_kwargs.keys())
        if overlap:
            raise ValueError(
                "estimator_kwargs cannot override managed parameters: "
                + ", ".join(sorted(overlap))
            )

        phi_A, phi_B, y = self._prepare_data(scores_A, scores_B, preferences)
        n = len(y)

        C = 1.0 / max(self.regularization, 1e-12)
        logger.info(
            "Fitting Bradley-Terry model: %d samples (ties removed), %d features, "
            "C=%.2f (lambda=%.4f), tie_epsilon=%.3f",
            n,
            self.k,
            C,
            self.regularization,
            self.tie_epsilon,
        )

        if n < 2:
            logger.warning("Too few samples (%d) to fit BT model. Using uniform weights.", n)
            self.weights = np.ones(self.k) / max(self.k, 1)
            self.intercept = 0.0
            self._model = None
            return self

        X_combined = np.vstack([phi_A, phi_B])
        self._scaler.fit(X_combined)

        phi_A_scaled = self._scaler.transform(phi_A)
        phi_B_scaled = self._scaler.transform(phi_B)
        X_diff = phi_A_scaled - phi_B_scaled

        if self.estimator is not None:
            if estimator_kwargs:
                logger.warning(
                    "Ignoring estimator_kwargs because a custom estimator instance was provided."
                )
            self._model = self.estimator
        else:
            lr_kwargs = {
                "C": C,
                "fit_intercept": fit_intercept,
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
            }
            lr_kwargs.update(self.default_estimator_kwargs)
            lr_kwargs.update(estimator_kwargs)
            self._model = LogisticRegression(**lr_kwargs)

        self._model.fit(X_diff, y)

        if hasattr(self._model, "coef_"):
            self.weights = np.asarray(self._model.coef_[0], dtype=float)
            self.intercept = float(self._model.intercept_[0]) if fit_intercept else 0.0
        elif hasattr(self._model, "feature_importances_"):
            self.weights = np.asarray(self._model.feature_importances_, dtype=float)
            self.intercept = 0.0
        else:
            logger.warning("Cannot extract weights from estimator %s", type(self._model))
            self.weights = np.zeros(self.k, dtype=float)
            self.intercept = 0.0

        if verbose:
            logger.info("Fitted weights: %s", self.weight_dict())
            logger.info("Intercept: %.4f", self.intercept)
            if hasattr(self._model, "score"):
                try:
                    train_acc = self._model.score(X_diff, y)
                    logger.info("Training accuracy: %.2f%%", train_acc * 100)
                except Exception:
                    logger.debug("Could not compute training accuracy", exc_info=True)

        return self

    def predict_proba(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
    ) -> np.ndarray:
        """Predict probability that A is preferred over B."""
        assert self._model is not None, "Model not fitted yet. Call .fit() first."
        phi_A, phi_B, _ = self._prepare_data(scores_A, scores_B)

        phi_A_scaled = self._scaler.transform(phi_A)
        phi_B_scaled = self._scaler.transform(phi_B)
        X_diff = phi_A_scaled - phi_B_scaled

        proba = self._model.predict_proba(X_diff)
        class_1_idx = list(self._model.classes_).index(1.0)
        return proba[:, class_1_idx]

    def predict(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict preference labels: ``1 = prefer A``, ``0 = prefer B``."""
        proba = self.predict_proba(scores_A, scores_B)
        return (proba >= threshold).astype(int)

    def weight_dict(self) -> dict[str, float]:
        """Return weights as ``{dimension_name: weight}``."""
        if self.weights is None:
            return {}
        return {name: float(self.weights[i]) for i, name in enumerate(self.dimension_names)}
