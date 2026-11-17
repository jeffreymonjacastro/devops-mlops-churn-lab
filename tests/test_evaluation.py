"""Controles de costo, umbral y segmentos de la fase 4."""
import numpy as np
import pandas as pd
import pytest

from evaluate_and_register import classification_metrics, expected_cost, slice_metrics


def test_expected_cost_counts_false_negatives_and_false_positives():
    # Un FN y un FP entre cuatro clientes: (200 + 100) / 4.
    assert expected_cost([0, 0, 1, 1], [0, 1, 0, 1], 200, 100) == 75


def test_lower_threshold_recovers_churners():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    assert classification_metrics(y, probabilities, 0.5)["recall"] == 0.5
    assert classification_metrics(y, probabilities, 0.3)["recall"] == 1.0


def test_slice_recall_uses_positions_even_with_nonconsecutive_index():
    test = pd.DataFrame(
        {"gender": ["Male", "Female", "Male", "Female"], "churn": [1, 1, 0, 1]},
        index=[10, 20, 30, 40],
    )
    cfg = {"data": {"target": "churn"}, "quality_gate": {"slices": ["gender"]}}
    result = slice_metrics(test, np.array([0.9, 0.8, 0.1, 0.2]), 0.5, cfg)
    assert result["gender"]["groups"]["Female"]["recall"] == 0.5
    assert result["gender"]["groups"]["Male"]["recall"] == 1.0
    assert result["gender"]["recall_gap"] == pytest.approx(0.5)
