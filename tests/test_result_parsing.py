from agent_eval.metrics import normalize_results


def test_normalize_results_adds_expected_columns():
    frame = normalize_results([{"model": "m1", "defense": "baseline"}])
    assert "scenario_id" in frame.columns
    assert "robust_success" in frame.columns
    assert len(frame) == 1
