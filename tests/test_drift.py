"""Tests for drift scoring and the published report."""

from __future__ import annotations

import pytest

from nodes.core.drift import (
    DRIFT_FLOOR,
    DriftReport,
    build_report,
    cosine_similarity,
    score_against_reference,
)


def test_cosine_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_scales_invariantly():
    assert cosine_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_cosine_zero_vector_returns_zero_not_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_score_against_reference_returns_one_per_frame():
    ref = [1.0, 1.0]
    frames = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    scores = score_against_reference(ref, frames)
    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0)


def test_report_aggregates():
    report = DriftReport(scores=[0.95, 0.93, 0.91], seqs=[1, 2, 3])
    assert report.n == 3
    assert report.mean == pytest.approx(0.93)
    assert report.worst == pytest.approx(0.91)
    assert report.worst_seq == 3
    assert report.holding is True


def test_report_detects_failure_to_hold():
    report = DriftReport(scores=[0.95, 0.70, 0.88], seqs=[1, 2, 3])
    assert report.holding is False
    assert report.worst_seq == 2
    assert report.degradation_seq() == 2


def test_slope_is_negative_when_degrading():
    report = DriftReport(scores=[0.95, 0.90, 0.85], seqs=[1, 2, 3])
    assert report.slope() < 0


def test_slope_is_zero_for_a_single_shot():
    assert DriftReport(scores=[0.9], seqs=[1]).slope() == 0.0


def test_empty_report_does_not_claim_success():
    report = DriftReport(scores=[])
    assert report.holding is False
    assert report.mean == 0.0
    assert report.worst_seq is None


def test_lift_over_baseline():
    report = DriftReport(
        scores=[0.95, 0.94],
        seqs=[1, 2],
        baseline=[0.70, 0.72],
    )
    assert report.lift() == pytest.approx(0.945 - 0.71)


def test_lift_is_none_without_baseline():
    assert DriftReport(scores=[0.9], seqs=[1]).lift() is None


def test_build_report_full_pipeline():
    ref = [1.0, 0.0]
    frames = [[1.0, 0.0], [0.9, 0.1]]
    report = build_report(ref, frames)
    assert report.n == 2
    assert report.seqs == [1, 2]
    assert report.mean > 0.9


def test_build_report_with_baseline():
    ref = [1.0, 0.0]
    frames = [[1.0, 0.0]]
    baseline = [[0.0, 1.0]]
    report = build_report(ref, frames, baseline_frames=baseline)
    assert report.baseline is not None
    assert report.lift() == pytest.approx(1.0)


def test_to_dict_shape():
    report = DriftReport(scores=[0.95, 0.93], seqs=[1, 2], label="locked")
    payload = report.to_dict()
    assert payload["label"] == "locked"
    assert payload["shots"] == 2
    assert payload["holding"] is True
    assert len(payload["curve"]) == 2
    assert payload["curve"][0] == {"seq": 1, "score": 0.95}


def test_markdown_report_contains_the_evidence():
    report = DriftReport(scores=[0.95, 0.70], seqs=[1, 2])
    md = report.markdown()
    assert "# Drift report" in md
    assert "Mean similarity" in md
    assert "| shot | similarity |" in md
    assert "**drift**" in md  # the failing shot is flagged
    assert "First shot below floor" in md


def test_markdown_reports_lift_when_baseline_present():
    report = DriftReport(scores=[0.95], seqs=[1], baseline=[0.60])
    md = report.markdown()
    assert "Lift from locking" in md
    assert "+0.350" in md


def test_ascii_curve_renders():
    report = DriftReport(scores=[0.95, 0.93, 0.91], seqs=[1, 2, 3])
    curve = report.ascii_curve()
    assert "drift curve" in curve
    assert "*" in curve


def test_ascii_curve_handles_empty():
    assert DriftReport(scores=[]).ascii_curve() == "(no scores)"


def test_floor_is_a_sane_similarity():
    assert 0 < DRIFT_FLOOR < 1
