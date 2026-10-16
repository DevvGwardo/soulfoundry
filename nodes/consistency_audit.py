"""ConsistencyAudit: the differentiator.

Every competitor claims "the same face across every scene" and attaches no
number. This node emits the number: per-shot similarity against the locked
reference, a drift curve, and the shot where it starts to fail.

The honest framing is the product. A measured partial failure is a stronger
result than an unmeasured claim, and it is what makes this node worth demoing.
"""

from __future__ import annotations

import json

from .core.drift import DRIFT_FLOOR, DriftReport, build_report
from .core.types import DRIFT_REPORT, ELEMENT

CATEGORY = "SoulFoundry/evidence"


class ConsistencyAudit:
    """Score generated frames against a locked reference."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "mode": (["embeddings", "scores"], {"default": "embeddings"}),
            },
            "optional": {
                "element": (ELEMENT,),
                "reference_json": ("STRING", {"default": "[]", "multiline": True}),
                "frames_json": ("STRING", {"default": "[]", "multiline": True}),
                "scores_json": ("STRING", {"default": "[]", "multiline": True}),
                "baseline_json": ("STRING", {"default": "[]", "multiline": True}),
                "seqs_json": ("STRING", {"default": "[]", "multiline": True}),
                "label": ("STRING", {"default": "locked", "multiline": False}),
                "project_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (DRIFT_REPORT, "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("report", "mean_similarity", "holding", "markdown")
    FUNCTION = "audit"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Per-shot face-similarity scoring against a locked Element, with a "
        "drift curve, trend slope, and the first shot that drops below the "
        "floor. mode='embeddings' computes cosine similarity; mode='scores' "
        "takes pre-computed per-shot scores."
    )

    def audit(self, mode: str = "embeddings", element: dict | None = None,
              reference_json: str = "[]", frames_json: str = "[]",
              scores_json: str = "[]", baseline_json: str = "[]",
              seqs_json: str = "[]", label: str = "locked",
              project_path: str = "") -> tuple[dict, float, bool, str]:
        seqs = _load_list(seqs_json, "seqs_json", cast=int)
        baseline = _load_list(baseline_json, "baseline_json") or None

        if mode == "scores":
            scores = _load_list(scores_json, "scores_json", cast=float)
            if not scores:
                raise ValueError("mode='scores' needs a non-empty scores_json")
            report = DriftReport(
                scores=scores,
                seqs=seqs or list(range(1, len(scores) + 1)),
                baseline=baseline,
                label=label,
            )
        else:
            reference = _load_list(reference_json, "reference_json", cast=float)
            if not reference:
                if element and element.get("embed_path"):
                    raise ValueError(
                        "element has an embed_path but no embedding was "
                        "supplied; load it and pass reference_json"
                    )
                raise ValueError("mode='embeddings' needs a reference_json vector")
            frames = _load_vectors(frames_json, "frames_json")
            if not frames:
                raise ValueError("mode='embeddings' needs a non-empty frames_json")
            report = build_report(
                reference,
                frames,
                seqs=seqs or None,
                baseline_frames=baseline,
                label=label,
            )

        if project_path:
            from .core.store import ProjectStore

            with ProjectStore(project_path) as store:
                for seq, score in zip(report.seqs, report.scores):
                    store.record_drift(seq, score)

        return (
            report.to_dict(),
            round(report.mean, 4),
            report.holding,
            report.markdown(),
        )


def _load_list(raw: str, field: str, cast=float) -> list:
    try:
        payload = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{field} must decode to an array")
    try:
        return [cast(x) for x in payload]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} has non-numeric entries: {exc}") from exc


def _load_vectors(raw: str, field: str) -> list[list[float]]:
    try:
        payload = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{field} must decode to an array of arrays")
    out: list[list[float]] = []
    for i, row in enumerate(payload):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{field}[{i}] must be a non-empty array")
        try:
            out.append([float(x) for x in row])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{i}] has non-numeric entries: {exc}") from exc
    return out


__all__ = ["ConsistencyAudit", "DRIFT_FLOOR"]
