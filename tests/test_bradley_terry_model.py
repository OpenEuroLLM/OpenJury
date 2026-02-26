import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from openjury.bradley_terry import FeatureBradleyTerry


def _synthetic_bt_data():
    scores_A = pd.DataFrame(
        {
            "adherence": [9, 8, 7, 2, 3, 4],
            "clarity": [8, 7, 8, 2, 3, 4],
        }
    )
    scores_B = pd.DataFrame(
        {
            "adherence": [2, 3, 4, 9, 8, 7],
            "clarity": [2, 3, 4, 8, 7, 8],
        }
    )
    # first half A wins, second half B wins
    preferences = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return scores_A, scores_B, preferences


def test_invalid_tie_epsilon_raises():
    with pytest.raises(ValueError):
        FeatureBradleyTerry(criterion_names=["x"], tie_epsilon=0.5)


def test_fit_learns_weights_on_synthetic_data():
    scores_A, scores_B, preferences = _synthetic_bt_data()
    bt = FeatureBradleyTerry(criterion_names=["adherence", "clarity"])

    bt.fit(scores_A, scores_B, preferences, verbose=False)

    assert bt.weights is not None
    assert len(bt.weights) == 2
    assert bt.num_criteria == 2
    assert bt._model is not None
    assert set(bt.weight_dict().keys()) == {"adherence", "clarity"}


def test_tie_epsilon_filters_near_ties():
    scores_A, scores_B, _ = _synthetic_bt_data()
    preferences = pd.Series([0.0, 0.49, 0.5, 0.51, 1.0, 1.0])

    bt_strict = FeatureBradleyTerry(
        criterion_names=["adherence", "clarity"], tie_epsilon=0.0
    )
    bt_loose = FeatureBradleyTerry(
        criterion_names=["adherence", "clarity"], tie_epsilon=0.05
    )

    _, _, y_strict = bt_strict._prepare_data(scores_A, scores_B, preferences)
    _, _, y_loose = bt_loose._prepare_data(scores_A, scores_B, preferences)

    assert y_strict is not None and y_loose is not None
    assert len(y_loose) < len(y_strict)


def test_fit_kwargs_forwarded_to_default_estimator():
    scores_A, scores_B, preferences = _synthetic_bt_data()
    bt = FeatureBradleyTerry(criterion_names=["adherence", "clarity"])

    bt.fit(scores_A, scores_B, preferences, verbose=False, max_iter=37)

    assert hasattr(bt._model, "max_iter")
    assert bt._model.max_iter == 37


def test_predict_proba_shape_and_range():
    scores_A, scores_B, preferences = _synthetic_bt_data()
    bt = FeatureBradleyTerry(criterion_names=["adherence", "clarity"])
    bt.fit(scores_A, scores_B, preferences, verbose=False)

    proba = bt.predict_proba(scores_A, scores_B)
    assert proba.shape == (len(scores_A),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_too_few_samples_falls_back_to_uniform_weights():
    scores_A = pd.DataFrame({"adherence": [8.0], "clarity": [8.0]})
    scores_B = pd.DataFrame({"adherence": [3.0], "clarity": [3.0]})
    preferences = pd.Series([0.0])

    bt = FeatureBradleyTerry(criterion_names=["adherence", "clarity"])
    bt.fit(scores_A, scores_B, preferences, verbose=False)

    assert bt.weights is not None
    assert np.allclose(bt.weights, np.array([0.5, 0.5]))
    assert bt.intercept == 0.0
