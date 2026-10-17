"""Drift scoring: the metric nobody else publishes.

Given per-shot similarity scores against a locked reference, produce a
publishable report. The honest framing matters here: a partial failure that is
measured is a stronger result than an unmeasured claim of perfection.

Pure Python. No numpy required, but uses it if present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Cosine similarity below this reads as "a different person".
DRIFT_FLOOR = 0.75
# Above this the shots are effectively the same face.
LOCK_CEILING = 0.90


@dataclass
class DriftReport:
    """Per-shot drift against a locked reference, plus the aggregate curve."""

    scores: list[float] = field(default_factory=list)
    seqs: list[int] = field(default_factory=list)
    baseline: list[float] | None = None
    label: str = "locked"
    #: Which embedder produced these scores. ``arcface`` means real face
    #: identity; anything else must be labelled in every artifact, because a
    #: perceptual descriptor is not a face match and must not be published as
    #: one.
    backend: str = ""
    notes: list[str] = field(default_factory=list)

    # -- derived ------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return sum(self.scores) / self.n if self.n else 0.0

    @property
    def worst(self) -> float:
        return min(self.scores) if self.scores else 0.0

    @property
    def worst_seq(self) -> int | None:
        if not self.scores:
            return None
        return self.seqs[self.scores.index(self.worst)]

    @property
    def holding(self) -> bool:
        """True when no shot has drifted below the floor."""
        return bool(self.scores) and self.worst >= DRIFT_FLOOR

    def slope(self) -> float:
        """Least-squares slope of score vs shot index.

        Negative means the face degrades across the sequence, which is the
        thing worth reporting honestly.
        """
        if self.n < 2:
            return 0.0
        xs = [float(i) for i in range(self.n)]
        ys = self.scores
        mx = sum(xs) / self.n
        my = sum(ys) / self.n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

    def degradation_seq(self, floor: float = DRIFT_FLOOR) -> int | None:
        """First shot that fell below the floor, if any."""
        for seq, score in zip(self.seqs, self.scores):
            if score < floor:
                return seq
        return None

    def lift(self) -> float | None:
        """Mean improvement over the unlocked baseline, if one was recorded."""
        if not self.baseline:
            return None
        base = sum(self.baseline) / len(self.baseline)
        return self.mean - base

    # -- output -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "shots": self.n,
            "mean": round(self.mean, 4),
            "worst": round(self.worst, 4),
            "worst_seq": self.worst_seq,
            "holding": self.holding,
            "slope": round(self.slope(), 5),
            "degradation_seq": self.degradation_seq(),
            "backend": self.backend,
            "is_face_embedding": self.backend == "arcface",
            "notes": self.notes,
            "baseline_mean": (
                round(sum(self.baseline) / len(self.baseline), 4)
                if self.baseline
                else None
            ),
            "lift": round(self.lift(), 4) if self.lift() is not None else None,
            "curve": [
                {"seq": s, "score": round(v, 4)}
                for s, v in zip(self.seqs, self.scores)
            ],
        }

    def markdown(self) -> str:
        """The artifact that goes in the demo video."""
        lines = [
            f"# Drift report: {self.label}",
            "",
        ]
        if self.backend and self.backend != "arcface":
            lines += [
                f"> **Not a face-identity score.** These numbers come from the "
                f"`{self.backend}` descriptor, which has no notion of identity. "
                f"Do not publish them as a face-match result.",
                "",
            ]
        elif self.backend == "arcface":
            lines += [
                "> Embedder: ArcFace (SCRFD detection, 5-point alignment, 512-d).",
                "",
            ]
        lines += [
            f"- Shots scored: **{self.n}**",
            f"- Mean similarity: **{self.mean:.3f}**",
            f"- Worst shot: **{self.worst:.3f}** (shot {self.worst_seq})",
            f"- Holding (no shot below {DRIFT_FLOOR}): "
            f"**{'yes' if self.holding else 'no'}**",
            f"- Trend per shot: **{self.slope():+.5f}** "
            f"({'degrading' if self.slope() < -0.001 else 'stable'})",
        ]
        deg = self.degradation_seq()
        if deg is not None:
            lines.append(f"- First shot below floor: **{deg}**")
        if self.baseline:
            lines.append(
                f"- Unlocked baseline mean: "
                f"**{sum(self.baseline) / len(self.baseline):.3f}**"
            )
            lift = self.lift() or 0.0
            lines.append(f"- **Lift from locking: {lift:+.3f}**")
        lines += ["", "| shot | similarity |", "|---|---|"]
        for seq, score in zip(self.seqs, self.scores):
            flag = "" if score >= DRIFT_FLOOR else " **drift**"
            lines.append(f"| {seq} | {score:.3f}{flag} |")
        if self.notes:
            lines += ["", "## Notes", ""]
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)

    def ascii_curve(self, width: int = 48, height: int = 8) -> str:
        """Terminal curve, for logs and the demo recording."""
        if not self.scores:
            return "(no scores)"
        lo = min(self.scores)
        hi = max(self.scores)
        span = (hi - lo) or 1.0
        rows: list[list[str]] = [[" "] * width for _ in range(height)]
        for i, score in enumerate(self.scores):
            x = min(width - 1, int(i * (width - 1) / max(1, self.n - 1)))
            y = height - 1 - int((score - lo) / span * (height - 1))
            rows[y][x] = "*"
        floor_y = height - 1 - int((DRIFT_FLOOR - lo) / span * (height - 1))
        if 0 <= floor_y < height:
            for x in range(width):
                if rows[floor_y][x] == " ":
                    rows[floor_y][x] = "-"
        header = f"drift curve  min {lo:.3f}  max {hi:.3f}  floor {DRIFT_FLOOR}"
        return "\n".join([header] + ["".join(r) for r in rows])


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two embedding vectors.

    Returns 0.0 for a zero vector rather than raising, because a failed
    embedding extraction should read as "no similarity", not crash a batch.
    """
    if len(a) != len(b):
        raise ValueError(f"embedding length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def score_against_reference(
    reference: Sequence[float],
    frames: Iterable[Sequence[float]],
) -> list[float]:
    """Similarity of each frame embedding against the locked reference."""
    return [cosine_similarity(reference, f) for f in frames]


def build_report(
    reference: Sequence[float],
    frames: Sequence[Sequence[float]],
    *,
    seqs: Sequence[int] | None = None,
    baseline_frames: Sequence[Sequence[float]] | None = None,
    label: str = "locked",
) -> DriftReport:
    """Full pipeline: embeddings in, publishable report out."""
    scores = score_against_reference(reference, frames)
    baseline = (
        score_against_reference(reference, baseline_frames)
        if baseline_frames
        else None
    )
    return DriftReport(
        scores=scores,
        seqs=list(seqs) if seqs is not None else list(range(1, len(scores) + 1)),
        baseline=baseline,
        label=label,
    )
