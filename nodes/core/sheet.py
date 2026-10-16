"""Character sheet: the locking artifact.

A 3-panel turnaround on a plain gray backdrop. The lighting rules are the
part that decides whether a sheet is usable, so they are baked into the
template rather than left to the user.

QC is deliberately split into two halves:
  * ``qc_sheet_metrics`` - pure math over already-computed image statistics.
    Testable with no image library and no GPU.
  * ``qc_sheet`` - optional path that decodes an image with Pillow and numpy
    and feeds the metrics above. Imported lazily so the core stays
    dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

PANELS: tuple[str, ...] = (
    "full-body front, head to toe, feet and shoes in frame",
    "full-body back, identical pose and distance to the front panel",
    "close-up head and shoulders, front, eyes level with camera",
)

BACKDROP = "plain seamless mid-gray studio backdrop (#808080)"

LIGHTING_RULES: tuple[str, ...] = (
    "soft even near-shadowless key light at 5600K",
    "no glare or blown highlight on the hair",
    "visible catchlight in both eyes",
    "identical backdrop and identical lighting across all three panels",
    "feet visible in both full-body panels",
)

NEGATIVE = (
    "harsh directional shadow, blown highlights, lens flare, motion blur, "
    "text, watermark, logo, extra limbs, extra fingers, multiple faces per "
    "panel, cropped feet, inconsistent lighting between panels, "
    "different backdrop per panel"
)

SHEET_TEMPLATE = """\
Character reference sheet, {panel_count} panels side by side, single image.
Panels, left to right:
{panels}
{identity}
Backdrop: {backdrop}, identical in every panel.
Lighting: {lighting}
Style: {style}
The same person in all panels. Photorealistic, sharp focus, natural
proportions, neutral expression, arms relaxed at sides in the full-body panels.
Negative: {negative}
"""


def build_sheet_prompt(
    identity_prompt: str,
    *,
    style: str = "studio reference photography",
    panel_count: int = 3,
    backdrop: str = BACKDROP,
    extra_negative: str = "",
) -> str:
    """Compose the sheet prompt with the lighting rules baked in.

    ``identity_prompt`` is the output of ``compile_prompt`` - passed through
    verbatim, never rewritten.
    """
    if panel_count != 3:
        raise ValueError(
            "SoulFoundry ships the 3-panel turnaround only; "
            f"got panel_count={panel_count}"
        )
    panels = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(PANELS))
    negative = NEGATIVE + (f", {extra_negative}" if extra_negative.strip() else "")
    return SHEET_TEMPLATE.format(
        panel_count=panel_count,
        panels=panels,
        identity=identity_prompt.strip(),
        backdrop=backdrop,
        lighting="; ".join(LIGHTING_RULES) + ".",
        style=style,
        negative=negative,
    )


# --- QC ---------------------------------------------------------------------

# Thresholds. These are the "batch and pick" filter: a sheet that fails any of
# these is rejected before a human ever looks at it, which is what makes
# generating 8 and keeping 1 cheap.
THRESHOLDS: dict[str, float] = {
    "min_panels": 3,
    "max_shadow_ratio": 0.45,   # fraction of pixels below the mid-tone
    "min_highlight_clip": 0.0,  # blown-highlight fraction allowed in hair
    "max_highlight_clip": 0.02,
    "min_catchlight": 0.15,     # relative local contrast in the eye region
    "max_backdrop_delta": 12.0, # per-panel mean deviation, 0-255 scale
    "min_subject_coverage": 0.08,
    "max_subject_coverage": 0.80,
}


@dataclass
class SheetQCResult:
    """Outcome of the automated reject test."""

    passed: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "checks": self.checks,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "reasons": self.reasons,
        }

    def report(self) -> str:
        lines = [
            f"SHEET QC: {'PASS' if self.passed else 'REJECT'} "
            f"(score {self.score:.2f})",
        ]
        for name, ok in self.checks.items():
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}")
        for reason in self.reasons:
            lines.append(f"  -> {reason}")
        return "\n".join(lines)


def qc_sheet_metrics(metrics: Mapping[str, float]) -> SheetQCResult:
    """Grade already-computed image statistics.

    Expected keys (all optional; a missing key is skipped, not failed):
      panels             - number of panels detected
      shadow_ratio       - fraction of pixels in shadow
      highlight_clip     - fraction of blown-highlight pixels
      catchlight         - relative eye-region local contrast, 0..1
      backdrop_delta     - max per-panel mean deviation, 0..255
      subject_coverage   - fraction of frame occupied by the subject
    """
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    if "panels" in metrics:
        ok = metrics["panels"] >= THRESHOLDS["min_panels"]
        checks["panel_count"] = ok
        if not ok:
            reasons.append(
                f"expected {int(THRESHOLDS['min_panels'])} panels, "
                f"found {int(metrics['panels'])}"
            )

    if "shadow_ratio" in metrics:
        ok = metrics["shadow_ratio"] <= THRESHOLDS["max_shadow_ratio"]
        checks["shadow_ratio"] = ok
        if not ok:
            reasons.append(
                f"shadows too harsh ({metrics['shadow_ratio']:.2f} > "
                f"{THRESHOLDS['max_shadow_ratio']}); relight near-shadowless"
            )

    if "highlight_clip" in metrics:
        ok = (
            THRESHOLDS["min_highlight_clip"]
            <= metrics["highlight_clip"]
            <= THRESHOLDS["max_highlight_clip"]
        )
        checks["hair_glare"] = ok
        if not ok:
            reasons.append(
                f"hair glare / blown highlights ({metrics['highlight_clip']:.3f})"
            )

    if "catchlight" in metrics:
        ok = metrics["catchlight"] >= THRESHOLDS["min_catchlight"]
        checks["catchlight"] = ok
        if not ok:
            reasons.append(
                f"no visible catchlight in eyes ({metrics['catchlight']:.2f})"
            )

    if "backdrop_delta" in metrics:
        ok = metrics["backdrop_delta"] <= THRESHOLDS["max_backdrop_delta"]
        checks["consistent_backdrop"] = ok
        if not ok:
            reasons.append(
                f"backdrop differs between panels "
                f"({metrics['backdrop_delta']:.1f} > "
                f"{THRESHOLDS['max_backdrop_delta']})"
            )

    if "subject_coverage" in metrics:
        ok = (
            THRESHOLDS["min_subject_coverage"]
            <= metrics["subject_coverage"]
            <= THRESHOLDS["max_subject_coverage"]
        )
        checks["subject_coverage"] = ok
        if not ok:
            reasons.append(
                f"subject coverage {metrics['subject_coverage']:.2f} outside "
                f"{THRESHOLDS['min_subject_coverage']}.."
                f"{THRESHOLDS['max_subject_coverage']} (feet likely cropped)"
            )

    total = len(checks) or 1
    score = sum(1 for ok in checks.values() if ok) / total
    return SheetQCResult(
        passed=all(checks.values()) and bool(checks),
        score=score,
        checks=checks,
        metrics=dict(metrics),
        reasons=reasons,
    )


def qc_sheet(image: Any, panels: int = 3) -> SheetQCResult:
    """Grade a real image. Requires Pillow + numpy (imported lazily).

    Splits the sheet into ``panels`` vertical strips, measures shadow ratio,
    highlight clipping, backdrop consistency and subject coverage, and
    estimates catchlight as local contrast in the upper-eye band of the
    close-up panel.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "qc_sheet(image=...) needs Pillow and numpy; "
            "use qc_sheet_metrics() for a dependency-free path"
        ) from exc

    if isinstance(image, (str, bytes)):
        img = Image.open(image)
    else:
        img = image
    img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32)

    height, width = arr.shape
    if width < panels:
        return SheetQCResult(
            passed=False,
            score=0.0,
            checks={"panel_count": False},
            reasons=[f"image too narrow for {panels} panels"],
        )

    strips = np.array_split(arr, panels, axis=1)
    metrics: dict[str, float] = {"panels": float(panels)}

    metrics["shadow_ratio"] = float((arr < 64).mean())
    metrics["highlight_clip"] = float((arr > 250).mean())

    means = [float(s.mean()) for s in strips]
    metrics["backdrop_delta"] = float(max(means) - min(means))

    # Subject coverage: pixels that differ from the backdrop's modal tone.
    backdrop_tone = float(np.median(arr))
    metrics["subject_coverage"] = float((np.abs(arr - backdrop_tone) > 24).mean())

    # Catchlight: local contrast in the eye band of the rightmost panel.
    closeup = strips[-1]
    band = closeup[int(closeup.shape[0] * 0.10) : int(closeup.shape[0] * 0.35)]
    if band.size:
        grad = np.abs(np.diff(band, axis=1))
        metrics["catchlight"] = float(min(1.0, grad.mean() / 24.0))
    _ = height
    return qc_sheet_metrics(metrics)


def pick_best(results: Sequence[SheetQCResult]) -> int:
    """Index of the highest-scoring sheet in a batch, ties broken by order."""
    if not results:
        raise ValueError("pick_best needs at least one result")
    return max(range(len(results)), key=lambda i: (results[i].passed, results[i].score, -i))
