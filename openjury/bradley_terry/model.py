"""Bradley-Terry model augmented with interpretable features.

Implements Eq. 2 from the proposal::

    p*(y1 ≻ y2 | x) = σ(wᵀ [Φ(x, y1) - Φ(x, y2)])

where Φ(x, y) is a vector of rubric scores and w are learnable weights.

Uses sklearn estimators (LogisticRegression by default) for robust fitting
with proper regularization, convergence, and feature scaling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from openjury._logging import logger


@dataclass
class FeatureBradleyTerry:
    """Bradley-Terry model with interpretable feature weights.

    Wraps an sklearn estimator (default: LogisticRegression) that operates
    on standardized rubric-score differences Φ(A) - Φ(B).

    Preference convention used by all scorers:
        0.0 = A wins, 0.5 = tie, 1.0 = B wins.

    Internally, ties are removed before fitting and labels are flipped so
    that y=1 means "A preferred" (consistent with delta_phi = phi_A - phi_B
    being positive when A scores higher).

    Attributes:
        dimension_names: Names of the rubric dimensions (features).
        regularization: L2 regularization strength λ (sklearn C = 1/λ).
        estimator: Optional custom sklearn estimator.
        weights: Learned weight vector w ∈ R^k (None before fitting).
        intercept: Learned bias term.
        fit_history: Kept for backward compatibility (empty for sklearn).
    """

    dimension_names: list[str]
    regularization: float = 0.01
    estimator: Any = None
    weights: np.ndarray | None = None
    intercept: float = 0.0
    fit_history: list[float] = field(default_factory=list)

    # Internal (not serialized)
    _scaler: StandardScaler = field(default_factory=StandardScaler, repr=False)
    _model: Any = field(default=None, repr=False)

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
            preferences: 0.0 = A wins, 1.0 = B wins, 0.5 = tie.
                        If None, only returns phi matrices (for prediction).

        Returns:
            phi_A, phi_B: Raw feature arrays ``(n, k)``.
            y: Binary labels (1 = A wins) with ties removed, or None.
        """
        phi_A = scores_A[self.dimension_names].values.astype(float)
        phi_B = scores_B[self.dimension_names].values.astype(float)

        if preferences is None:
            return phi_A, phi_B, None

        y_raw = preferences.values.astype(float)

        # Remove rows with NaN in features or labels
        valid = ~(
            np.isnan(phi_A).any(axis=1)
            | np.isnan(phi_B).any(axis=1)
            | np.isnan(y_raw)
        )
        n_dropped_nan = int((~valid).sum())
        if n_dropped_nan > 0:
            logger.warning(
                "Dropped %d/%d rows with NaN values", n_dropped_nan, len(valid)
            )
        phi_A = phi_A[valid]
        phi_B = phi_B[valid]
        y_raw = y_raw[valid]

        # Remove ties (y ≈ 0.5) — they degrade BCE fitting
        not_tie = np.abs(y_raw - 0.5) > 0.05
        n_ties = int((~not_tie).sum())
        if n_ties > 0:
            logger.info("Removed %d ties from %d samples", n_ties, len(y_raw))
        phi_A = phi_A[not_tie]
        phi_B = phi_B[not_tie]
        y_raw = y_raw[not_tie]

        # Convert: scorer convention 0.0 = A wins  →  y=1 (A wins)
        # This makes y=1 align with positive delta_phi (A scores higher)
        y = (y_raw < 0.5).astype(float)

        return phi_A, phi_B, y

    def fit(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
        preferences: pd.Series,
        lr: float = 0.01,  # ignored, kept for API compat
        n_steps: int = 1000,  # ignored, kept for API compat
        fit_intercept: bool = True,
        verbose: bool = True,
    ) -> FeatureBradleyTerry:
        """Fit the model on rubric score differences.

        Args:
            scores_A: Rubric scores for model A (one row per comparison).
            scores_B: Rubric scores for model B.
            preferences: 0.0 = A wins, 1.0 = B wins, 0.5 = tie.
                        Ties are removed before fitting.
            lr: *Ignored* — kept for backward compatibility.
            n_steps: *Ignored* — kept for backward compatibility.
            fit_intercept: Whether to fit a bias term.
            verbose: Whether to log details.

        Returns:
            ``self`` (fitted model) for method chaining.
        """
        phi_A, phi_B, y = self._prepare_data(scores_A, scores_B, preferences)
        n = len(y)

        C = 1.0 / max(self.regularization, 1e-12)
        logger.info(
            "Fitting Bradley-Terry model: %d samples (ties removed), "
            "%d features, C=%.2f (λ=%.4f)",
            n, self.k, C, self.regularization,
        )

        if n < 2:
            logger.warning(
                "Too few samples (%d) to fit BT model. Using uniform weights.", n
            )
            self.weights = np.ones(self.k) / self.k
            self.intercept = 0.0
            return self

        # Fit scaler on combined features
        X_combined = np.vstack([phi_A, phi_B])
        self._scaler.fit(X_combined)

        phi_A_scaled = self._scaler.transform(phi_A)
        phi_B_scaled = self._scaler.transform(phi_B)
        X_diff = phi_A_scaled - phi_B_scaled

        # Build estimator
        if self.estimator is not None:
            self._model = self.estimator
        else:
            self._model = LogisticRegression(
                C=C,
                fit_intercept=fit_intercept,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )

        self._model.fit(X_diff, y)

        # Extract weights (in scaled feature space)
        if hasattr(self._model, "coef_"):
            self.weights = self._model.coef_[0]
            self.intercept = (
                float(self._model.intercept_[0]) if fit_intercept else 0.0
            )
        elif hasattr(self._model, "feature_importances_"):
            self.weights = self._model.feature_importances_
            self.intercept = 0.0
        else:
            logger.warning(
                "Cannot extract weights from estimator %s", type(self._model)
            )
            self.weights = np.zeros(self.k)

        if verbose:
            logger.info("Fitted weights: %s", self.weight_dict())
            logger.info("Intercept: %.4f", self.intercept)
            train_acc = self._model.score(X_diff, y)
            logger.info("Training accuracy: %.2f%%", train_acc * 100)

        return self

    def predict_proba(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
    ) -> np.ndarray:
        """Predict probability that A is preferred over B.

        Args:
            scores_A: Rubric scores for model A.
            scores_B: Rubric scores for model B.

        Returns:
            ``(n,)`` array of probabilities p(A ≻ B | x).

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        assert self._model is not None, "Model not fitted yet. Call .fit() first."
        phi_A, phi_B, _ = self._prepare_data(scores_A, scores_B)

        phi_A_scaled = self._scaler.transform(phi_A)
        phi_B_scaled = self._scaler.transform(phi_B)
        X_diff = phi_A_scaled - phi_B_scaled

        # predict_proba returns [[P(class=0), P(class=1)]]
        # class 1 = A wins
        proba = self._model.predict_proba(X_diff)
        class_1_idx = list(self._model.classes_).index(1.0)
        return proba[:, class_1_idx]

    def predict(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict preference: 1 = prefer A, 0 = prefer B.

        Args:
            scores_A: Rubric scores for model A.
            scores_B: Rubric scores for model B.
            threshold: Decision boundary. Default 0.5.

        Returns:
            ``(n,)`` integer array of predictions.
        """
        proba = self.predict_proba(scores_A, scores_B)
        return (proba >= threshold).astype(int)

    def weight_dict(self) -> dict[str, float]:
        """Return weights as ``{dimension_name: weight}``."""
        if self.weights is None:
            return {}
        return {
            name: float(self.weights[i])
            for i, name in enumerate(self.dimension_names)
        }

    # ─────────────────────────────────────────────────────────────
    #  Bootstrap confidence intervals
    # ─────────────────────────────────────────────────────────────

    def bootstrap_weights(
        self,
        scores_A: pd.DataFrame,
        scores_B: pd.DataFrame,
        preferences: pd.Series,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
        fit_intercept: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Compute bootstrap confidence intervals for feature weights.

        Resamples instruction-level comparisons **with replacement**,
        refits the BT logistic regression on each bootstrap sample, and
        reports percentile-based confidence intervals.

        Call ``.fit()`` first — this method reuses the scaler fitted on
        the full dataset so that weight magnitudes are comparable across
        bootstrap iterations.

        Args:
            scores_A: Rubric scores for model A (one row per comparison).
            scores_B: Rubric scores for model B.
            preferences: 0.0 = A wins, 1.0 = B wins, 0.5 = tie.
            n_bootstrap: Number of bootstrap iterations.
            ci: Confidence level (default 0.95 → 95% CI).
            seed: Random seed for reproducibility.
            fit_intercept: Whether to fit a bias term.

        Returns:
            Dictionary mapping each dimension name (plus ``"_intercept"``)
            to a stats dict::

                {
                    "weight": float,       # point estimate (from full data)
                    "mean": float,         # bootstrap mean
                    "std": float,          # bootstrap std
                    "ci_lower": float,     # lower CI bound
                    "ci_upper": float,     # upper CI bound
                    "significant": bool,   # True if CI excludes zero
                }
        """
        # Prepare data (removes NaN rows and ties)
        phi_A, phi_B, y = self._prepare_data(scores_A, scores_B, preferences)
        n = len(y)

        if n < 10:
            logger.warning(
                "Too few decisive samples (%d) for bootstrap — need ≥10.", n,
            )
            return {}

        # Use the scaler fitted on the full data (from .fit())
        if not hasattr(self._scaler, "mean_"):
            X_combined = np.vstack([phi_A, phi_B])
            self._scaler.fit(X_combined)

        phi_A_s = self._scaler.transform(phi_A)
        phi_B_s = self._scaler.transform(phi_B)
        X_diff = phi_A_s - phi_B_s

        C = 1.0 / max(self.regularization, 1e-12)
        rng = np.random.RandomState(seed)

        boot_weights = np.zeros((n_bootstrap, self.k))
        boot_intercepts = np.zeros(n_bootstrap)
        n_failed = 0

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_b = X_diff[idx]
            y_b = y[idx]

            # Skip if bootstrap sample has only one class
            if len(np.unique(y_b)) < 2:
                n_failed += 1
                boot_weights[b] = np.nan
                boot_intercepts[b] = np.nan
                continue

            lr = LogisticRegression(
                C=C,
                fit_intercept=fit_intercept,
                max_iter=1000,
                solver="lbfgs",
                random_state=b,
            )
            lr.fit(X_b, y_b)
            boot_weights[b] = lr.coef_[0]
            boot_intercepts[b] = (
                float(lr.intercept_[0]) if fit_intercept else 0.0
            )

        if n_failed > 0:
            logger.info(
                "Bootstrap: %d/%d samples had single-class labels (skipped)",
                n_failed, n_bootstrap,
            )

        alpha = (1 - ci) / 2
        results: dict[str, dict[str, float]] = {}

        for i, dim in enumerate(self.dimension_names):
            w_samples = boot_weights[:, i]
            w_valid = w_samples[~np.isnan(w_samples)]

            if len(w_valid) < 10:
                results[dim] = {
                    "weight": float(self.weights[i]) if self.weights is not None else 0.0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "significant": False,
                }
                continue

            lo = float(np.percentile(w_valid, alpha * 100))
            hi = float(np.percentile(w_valid, (1 - alpha) * 100))

            results[dim] = {
                "weight": float(self.weights[i]) if self.weights is not None else 0.0,
                "mean": float(np.mean(w_valid)),
                "std": float(np.std(w_valid)),
                "ci_lower": lo,
                "ci_upper": hi,
                "significant": (lo > 0) or (hi < 0),  # CI excludes zero
            }

        # Intercept bootstrap stats
        int_valid = boot_intercepts[~np.isnan(boot_intercepts)]
        if len(int_valid) >= 10:
            lo = float(np.percentile(int_valid, alpha * 100))
            hi = float(np.percentile(int_valid, (1 - alpha) * 100))
            results["_intercept"] = {
                "weight": self.intercept,
                "mean": float(np.mean(int_valid)),
                "std": float(np.std(int_valid)),
                "ci_lower": lo,
                "ci_upper": hi,
                "significant": (lo > 0) or (hi < 0),
            }

        logger.info(
            "Bootstrap: %d iterations, %d succeeded, CI=%.0f%%",
            n_bootstrap, n_bootstrap - n_failed, ci * 100,
        )
        return results

    def save(self, path: str | Path) -> None:
        """Save fitted model to JSON.

        Saves weights, intercept, metadata, and scaler parameters so the
        model can be loaded for prediction without re-fitting.

        Args:
            path: Output file path.
        """
        path = Path(path)
        data = {
            "dimension_names": self.dimension_names,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "intercept": self.intercept,
            "regularization": self.regularization,
            "fit_history": self.fit_history,
            "scaler_mean": (
                self._scaler.mean_.tolist()
                if hasattr(self._scaler, "mean_")
                else None
            ),
            "scaler_scale": (
                self._scaler.scale_.tolist()
                if hasattr(self._scaler, "scale_")
                else None
            ),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved BT model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> FeatureBradleyTerry:
        """Load a fitted model from JSON.

        Reconstructs the scaler and creates a minimal LogisticRegression
        for ``predict_proba`` without needing the training data.

        Args:
            path: Path to a previously saved model JSON file.

        Returns:
            A ``FeatureBradleyTerry`` instance with loaded weights.
        """
        with open(path) as f:
            data = json.load(f)

        model = cls(
            dimension_names=data["dimension_names"],
            regularization=data.get("regularization", 0.01),
        )
        model.weights = (
            np.array(data["weights"]) if data["weights"] is not None else None
        )
        model.intercept = data.get("intercept", 0.0)
        model.fit_history = data.get("fit_history", [])

        # Restore scaler if saved
        if data.get("scaler_mean") is not None and data.get("scaler_scale") is not None:
            model._scaler = StandardScaler()
            model._scaler.mean_ = np.array(data["scaler_mean"])
            model._scaler.scale_ = np.array(data["scaler_scale"])
            model._scaler.var_ = model._scaler.scale_ ** 2
            model._scaler.n_features_in_ = len(data["scaler_mean"])

        # Build a minimal LogisticRegression for predict_proba
        if model.weights is not None:
            k = len(model.weights)
            lr = LogisticRegression()
            lr.classes_ = np.array([0.0, 1.0])
            lr.coef_ = model.weights.reshape(1, k)
            lr.intercept_ = np.array([model.intercept])
            lr.n_features_in_ = k
            model._model = lr

        return model
