"""Bradley-Terry model with interpretable features.

Implements the augmented Bradley-Terry model from the proposal:

    p*(y1 ≻ y2 | x) = σ(wᵀ [Φ(x, y1) - Φ(x, y2)])

where Φ(x, y) is a vector of rubric scores and w are learnable weights that
reveal which quality dimensions are most predictive of preference.

Quick start::

    from openjury.bradley_terry import FeatureBradleyTerry

    bt = FeatureBradleyTerry(dimension_names=["fluency", "usefulness", "clarity"])
    bt.fit(scores_A, scores_B, preferences)
    print(bt.weight_dict())
"""

from openjury.bradley_terry.model import FeatureBradleyTerry
from openjury.bradley_terry.analysis import BTAnalysis

__all__ = ["FeatureBradleyTerry", "BTAnalysis"]
