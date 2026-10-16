"""Tests for the character sheet template and the QC reject test."""

from __future__ import annotations

import pytest

from nodes.core.param_grid import CharacterSpec, compile_prompt
from nodes.core.sheet import (
    LIGHTING_RULES,
    NEGATIVE,
    PANELS,
    THRESHOLDS,
    build_sheet_prompt,
    pick_best,
    qc_sheet_metrics,
)


def test_three_panels_defined():
    assert len(PANELS) == 3
    assert "front" in PANELS[0]
    assert "back" in PANELS[1]
    assert "close-up" in PANELS[2]


def test_lighting_rules_cover_the_known_failure_modes():
    joined = " ".join(LIGHTING_RULES).lower()
    assert "5600k" in joined
    assert "glare" in joined
    assert "catchlight" in joined
    assert "identical" in joined


def test_sheet_prompt_bakes_in_rules_and_identity():
    identity = compile_prompt(CharacterSpec(name="santiago", hair_color="Ginger"))
    prompt = build_sheet_prompt(identity)
    assert identity.strip() in prompt
    assert "5600K" in prompt
    assert "catchlight" in prompt
    assert "no glare" in prompt.lower() or "no glare" in prompt
    assert NEGATIVE.split(",")[0] in prompt
    assert prompt.count("1.") >= 1 and "3." in prompt


def test_sheet_prompt_rejects_wrong_panel_count():
    with pytest.raises(ValueError, match="3-panel"):
        build_sheet_prompt("identity", panel_count=2)


def test_extra_negative_is_appended():
    prompt = build_sheet_prompt("identity", extra_negative="sunglasses")
    assert prompt.rstrip().endswith("sunglasses") or "sunglasses" in prompt


# --- QC ---------------------------------------------------------------------


def _good_metrics() -> dict:
    return {
        "panels": 3,
        "shadow_ratio": 0.20,
        "highlight_clip": 0.005,
        "catchlight": 0.40,
        "backdrop_delta": 4.0,
        "subject_coverage": 0.30,
    }


def test_good_sheet_passes():
    result = qc_sheet_metrics(_good_metrics())
    assert result.passed is True
    assert result.score == 1.0
    assert result.reasons == []


def test_harsh_shadows_reject():
    metrics = _good_metrics() | {"shadow_ratio": 0.80}
    result = qc_sheet_metrics(metrics)
    assert result.passed is False
    assert result.checks["shadow_ratio"] is False
    assert any("shadows too harsh" in r for r in result.reasons)


def test_hair_glare_reject():
    metrics = _good_metrics() | {"highlight_clip": 0.09}
    result = qc_sheet_metrics(metrics)
    assert result.passed is False
    assert any("glare" in r for r in result.reasons)


def test_missing_catchlight_reject():
    metrics = _good_metrics() | {"catchlight": 0.02}
    result = qc_sheet_metrics(metrics)
    assert result.passed is False
    assert any("catchlight" in r for r in result.reasons)


def test_inconsistent_backdrop_reject():
    metrics = _good_metrics() | {"backdrop_delta": 40.0}
    result = qc_sheet_metrics(metrics)
    assert result.passed is False
    assert any("backdrop differs" in r for r in result.reasons)


def test_wrong_panel_count_reject():
    metrics = _good_metrics() | {"panels": 2}
    result = qc_sheet_metrics(metrics)
    assert result.passed is False
    assert any("expected 3 panels" in r for r in result.reasons)


def test_cropped_feet_reject_via_subject_coverage():
    metrics = _good_metrics() | {"subject_coverage": 0.95}
    result = qc_sheet_metrics(metrics)
    assert result.passed is False
    assert any("feet likely cropped" in r for r in result.reasons)


def test_empty_metrics_does_not_pass():
    """No checks ran, so this must not report success."""
    result = qc_sheet_metrics({})
    assert result.passed is False
    assert result.checks == {}


def test_score_is_fraction_of_checks_passed():
    metrics = _good_metrics() | {"shadow_ratio": 0.80, "catchlight": 0.0}
    result = qc_sheet_metrics(metrics)
    assert 0 < result.score < 1
    assert result.passed is False


def test_thresholds_are_sane():
    assert THRESHOLDS["min_panels"] == 3
    assert 0 < THRESHOLDS["max_shadow_ratio"] < 1
    assert 0 <= THRESHOLDS["min_highlight_clip"] < THRESHOLDS["max_highlight_clip"] < 1


def test_pick_best_prefers_passing_then_score():
    good = qc_sheet_metrics(_good_metrics())
    ok = qc_sheet_metrics(_good_metrics() | {"catchlight": 0.0})
    bad = qc_sheet_metrics(_good_metrics() | {"shadow_ratio": 0.9, "catchlight": 0.0})
    assert pick_best([bad, ok, good]) == 2
    assert pick_best([bad, ok]) == 1


def test_pick_best_requires_input():
    with pytest.raises(ValueError):
        pick_best([])


def test_qc_report_string_is_readable():
    result = qc_sheet_metrics(_good_metrics() | {"catchlight": 0.0})
    report = result.report()
    assert "REJECT" in report
    assert "catchlight" in report
    assert "->" in report


def test_result_serialises_to_dict():
    result = qc_sheet_metrics(_good_metrics())
    payload = result.to_dict()
    assert payload["passed"] is True
    assert isinstance(payload["score"], float)
    assert isinstance(payload["checks"], dict)
    assert "metrics" in payload and "reasons" in payload
