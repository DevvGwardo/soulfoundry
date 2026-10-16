"""SheetQC: the automated reject test.

This is what makes batch-and-pick cheap. Generate 8 sheets, auto-reject the
lighting failures, keep the one that passes.

Two paths:
  * ``mode="metrics"`` grades a JSON of image statistics (no dependencies).
  * ``mode="image"`` decodes an IMAGE input with Pillow/numpy.
"""

from __future__ import annotations

import json

from .core.sheet import SheetQCResult, pick_best, qc_sheet, qc_sheet_metrics
from .core.types import QC_REPORT

CATEGORY = "SoulFoundry/sheet"


class SheetQC:
    """Reject bad sheets before a human ever looks at them."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "mode": (["metrics", "image"], {"default": "metrics"}),
            },
            "optional": {
                "metrics_json": ("STRING", {"default": "{}", "multiline": True}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (QC_REPORT, "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("report", "passed", "score", "reasons")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Automated character sheet QC. Checks panel count, shadow ratio, hair "
        "glare, catchlight presence, backdrop consistency and subject coverage."
    )

    def run(self, mode: str = "metrics", metrics_json: str = "{}",
            image=None) -> tuple[dict, bool, float, str]:
        if mode == "image":
            if image is None:
                raise ValueError("mode='image' requires an IMAGE input")
            result = qc_sheet(image)
        else:
            try:
                metrics = json.loads(metrics_json or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"metrics_json is not valid JSON: {exc}") from exc
            if not isinstance(metrics, dict):
                raise ValueError("metrics_json must decode to an object")
            result = qc_sheet_metrics(metrics)
        return (
            result.to_dict(),
            result.passed,
            float(result.score),
            "\n".join(result.reasons),
        )


class SheetQCBatch:
    """Pick the best sheet from a batch of QC reports."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "reports_json": ("STRING", {"default": "[]", "multiline": True}),
            },
        }

    RETURN_TYPES = ("INT", QC_REPORT)
    RETURN_NAMES = ("best_index", "best_report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Choose the highest-scoring sheet from a batch of QC reports."

    def run(self, reports_json: str) -> tuple[int, dict]:
        try:
            payload = json.loads(reports_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"reports_json is not valid JSON: {exc}") from exc
        if not isinstance(payload, list) or not payload:
            raise ValueError("reports_json must decode to a non-empty array")
        results = [
            SheetQCResult(
                passed=bool(item.get("passed")),
                score=float(item.get("score", 0.0)),
                checks=item.get("checks", {}),
                metrics=item.get("metrics", {}),
                reasons=item.get("reasons", []),
            )
            for item in payload
        ]
        best = pick_best(results)
        return (best, results[best].to_dict())
