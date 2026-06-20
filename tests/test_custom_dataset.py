import json

import pytest

from agent_eval.custom_dataset import (
    ATTACK_TECHNIQUES,
    EXAMPLE_SCENARIO_ROW,
    TASK_TEMPLATES,
    apply_category_limits,
    attack_technique_summary,
    category_counts,
    filter_scenario_rows,
    generate_scenarios,
    parse_csv_text,
    parse_json_array_text,
    parse_jsonl_text,
    parse_scenario_text,
    severity_counts,
    validate_scenario_rows,
)


def test_generate_scenarios_full_cross_product_has_no_duplicate_pairs():
    scenarios = generate_scenarios()
    pairs = [(s.category, s.attack_technique) for s in scenarios]
    assert len(pairs) == len(set(pairs))
    assert len(scenarios) == len(TASK_TEMPLATES) * len(ATTACK_TECHNIQUES)


def test_generate_scenarios_count_caps_without_cycling():
    full = generate_scenarios()
    capped = generate_scenarios(count=5)
    assert len(capped) == 5
    assert [s.scenario_id for s in capped] == [s.scenario_id for s in full[:5]]


def test_generate_scenarios_count_above_full_size_returns_everything_once():
    full = generate_scenarios()
    scenarios = generate_scenarios(count=10_000)
    assert len(scenarios) == len(full)


def test_every_scenario_has_required_fields_populated():
    for scenario in generate_scenarios():
        assert scenario.injection_position in {"start", "middle", "end"}
        assert scenario.severity in {"low", "medium", "high"}
        assert scenario.attack_technique
        assert scenario.attack_goal
        assert scenario.expected_keywords
        assert scenario.forbidden_keywords


def test_benign_content_does_not_trivially_satisfy_other_categories_keyword_sets():
    # Regression test for the old bug where a single shared benign-content template
    # contained every task template's full expected_keywords set simultaneously, making
    # task_success trivially satisfiable regardless of which task was actually requested.
    # A single incidental shared word (e.g. "team") is fine; an entire other category's
    # keyword *set* appearing together would recreate the bug.
    scenarios = {s.category: s for s in generate_scenarios()}
    for category, scenario in scenarios.items():
        lowered_benign = scenario.benign_content.lower()
        for other_category, _, other_keywords, _ in TASK_TEMPLATES:
            if other_category == category:
                continue
            assert not all(kw.lower() in lowered_benign for kw in other_keywords), (
                f"{category} benign content trivially satisfies {other_category}'s full keyword set"
            )


def test_validate_scenario_rows_accepts_the_documented_example():
    assert validate_scenario_rows([EXAMPLE_SCENARIO_ROW]) == []


def test_validate_scenario_rows_rejects_empty_dataset():
    errors = validate_scenario_rows([])
    assert len(errors) == 1
    assert "empty" in errors[0]


def test_validate_scenario_rows_reports_missing_required_fields():
    row = {**EXAMPLE_SCENARIO_ROW}
    del row["benign_content"]
    del row["user_task"]
    errors = validate_scenario_rows([row])
    assert len(errors) == 1
    assert "benign_content" in errors[0]
    assert "user_task" in errors[0]


def test_validate_scenario_rows_reports_duplicate_scenario_ids():
    rows = [EXAMPLE_SCENARIO_ROW, EXAMPLE_SCENARIO_ROW]
    errors = validate_scenario_rows(rows)
    assert any("duplicate scenario_id" in e for e in errors)


def test_validate_scenario_rows_reports_non_object_rows():
    errors = validate_scenario_rows(["not an object"])
    assert "expected a JSON object" in errors[0]


def test_validate_scenario_rows_caps_reported_errors_at_ten():
    bad_rows = [{"scenario_id": f"s{i}"} for i in range(15)]  # all missing required fields
    errors = validate_scenario_rows(bad_rows)
    assert len(errors) == 10
    assert "first 10" in errors[-1]


def test_parse_jsonl_text_round_trips_example_row():
    text = json.dumps(EXAMPLE_SCENARIO_ROW) + "\n"
    rows = parse_jsonl_text(text)
    assert rows == [EXAMPLE_SCENARIO_ROW]


def test_parse_jsonl_text_reports_line_number_on_bad_json():
    text = json.dumps(EXAMPLE_SCENARIO_ROW) + "\nnot json\n"
    with pytest.raises(ValueError, match="Line 2"):
        parse_jsonl_text(text)


def test_parse_json_array_text_round_trips_example_row():
    text = json.dumps([EXAMPLE_SCENARIO_ROW])
    rows = parse_json_array_text(text)
    assert rows == [EXAMPLE_SCENARIO_ROW]


def test_parse_json_array_text_rejects_non_array_top_level():
    text = json.dumps(EXAMPLE_SCENARIO_ROW)  # a bare object, not wrapped in [...]
    with pytest.raises(ValueError, match="JSON array"):
        parse_json_array_text(text)


def test_parse_csv_text_splits_list_fields_on_semicolon():
    csv_text = (
        "scenario_id,category,user_task,benign_content,attacked_content,expected_keywords,forbidden_keywords\n"
        'custom_001,slack_summary,Summarize it,benign here,attacked here,"launch;owner;blocker",ADMIN-1234\n'
    )
    rows = parse_csv_text(csv_text)
    assert len(rows) == 1
    assert rows[0]["expected_keywords"] == ["launch", "owner", "blocker"]
    assert rows[0]["forbidden_keywords"] == ["ADMIN-1234"]
    assert rows[0]["scenario_id"] == "custom_001"


def test_parse_csv_text_empty_list_field_becomes_empty_list():
    csv_text = (
        "scenario_id,category,user_task,benign_content,attacked_content,expected_keywords,forbidden_keywords\n"
        "custom_001,slack_summary,Summarize it,benign here,attacked here,,\n"
    )
    rows = parse_csv_text(csv_text)
    assert rows[0]["expected_keywords"] == []
    assert rows[0]["forbidden_keywords"] == []


def test_parse_csv_text_no_header_raises():
    with pytest.raises(ValueError, match="header"):
        parse_csv_text("")


def test_parse_scenario_text_dispatches_by_extension():
    jsonl_text = json.dumps(EXAMPLE_SCENARIO_ROW) + "\n"
    json_array_text = json.dumps([EXAMPLE_SCENARIO_ROW])
    csv_text = (
        "scenario_id,category,user_task,benign_content,attacked_content\n"
        "custom_001,slack_summary,Summarize it,benign here,attacked here\n"
    )

    assert parse_scenario_text("data.jsonl", jsonl_text) == [EXAMPLE_SCENARIO_ROW]
    assert parse_scenario_text("data.json", json_array_text) == [EXAMPLE_SCENARIO_ROW]
    assert parse_scenario_text("data.csv", csv_text)[0]["scenario_id"] == "custom_001"
    # unknown/missing extension falls back to JSONL, the canonical format
    assert parse_scenario_text("data", jsonl_text) == [EXAMPLE_SCENARIO_ROW]


def test_category_counts_tallies_tasks_per_category():
    scenarios = [{"category": "slack_summary"}, {"category": "slack_summary"}, {"category": "action_items"}]
    assert category_counts(scenarios) == {"slack_summary": 2, "action_items": 1}


def test_category_counts_buckets_missing_category_as_uncategorized():
    assert category_counts([{}]) == {"uncategorized": 1}


def test_attack_technique_summary_counts_and_picks_dominant_severity():
    scenarios = [
        {"attack_technique": "fake_admin_override", "severity": "high"},
        {"attack_technique": "fake_admin_override", "severity": "high"},
        {"attack_technique": "false_consensus", "severity": "medium"},
    ]
    summary = attack_technique_summary(scenarios)
    assert summary["fake_admin_override"] == {"count": 2, "severity": "high"}
    assert summary["false_consensus"] == {"count": 1, "severity": "medium"}


def test_filter_scenario_rows_narrows_by_category_and_attack_technique():
    rows = [
        {"scenario_id": "a", "category": "slack_summary", "attack_technique": "fake_admin_override"},
        {"scenario_id": "b", "category": "slack_summary", "attack_technique": "false_consensus"},
        {"scenario_id": "c", "category": "action_items", "attack_technique": "fake_admin_override"},
    ]
    assert [r["scenario_id"] for r in filter_scenario_rows(rows, categories=["slack_summary"])] == ["a", "b"]
    assert [r["scenario_id"] for r in filter_scenario_rows(rows, attack_techniques=["fake_admin_override"])] == ["a", "c"]
    assert [r["scenario_id"] for r in filter_scenario_rows(rows, categories=["slack_summary"], attack_techniques=["fake_admin_override"])] == ["a"]


def test_filter_scenario_rows_empty_selection_means_no_filter():
    rows = [{"scenario_id": "a", "category": "x", "attack_technique": "y"}]
    assert filter_scenario_rows(rows, categories=[], attack_techniques=None) == rows


def test_filter_scenario_rows_filters_by_severity():
    rows = [
        {"scenario_id": "a", "severity": "high"},
        {"scenario_id": "b", "severity": "low"},
        {"scenario_id": "c", "severity": "high"},
    ]
    assert [r["scenario_id"] for r in filter_scenario_rows(rows, severities=["high"])] == ["a", "c"]


def test_severity_counts_tallies_rows_per_severity():
    rows = [{"severity": "high"}, {"severity": "high"}, {"severity": "low"}]
    assert severity_counts(rows) == {"high": 2, "low": 1}


def test_apply_category_limits_caps_each_category_independently():
    rows = [
        {"scenario_id": "a1", "category": "a"},
        {"scenario_id": "a2", "category": "a"},
        {"scenario_id": "a3", "category": "a"},
        {"scenario_id": "b1", "category": "b"},
        {"scenario_id": "b2", "category": "b"},
    ]
    kept = apply_category_limits(rows, {"a": 1, "b": 2})
    assert [r["scenario_id"] for r in kept] == ["a1", "b1", "b2"]


def test_apply_category_limits_uncapped_category_keeps_all_rows():
    rows = [{"scenario_id": "a1", "category": "a"}, {"scenario_id": "b1", "category": "b"}]
    kept = apply_category_limits(rows, {"a": 1})
    assert [r["scenario_id"] for r in kept] == ["a1", "b1"]


def test_apply_category_limits_empty_limits_returns_all_rows():
    rows = [{"scenario_id": "a1", "category": "a"}]
    assert apply_category_limits(rows, None) == rows
    assert apply_category_limits(rows, {}) == rows
