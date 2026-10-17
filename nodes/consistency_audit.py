"""ConsistencyAudit: the differentiator.

Every competitor claims "the same face across every scene" and attaches no
number. This node emits the number: per-shot similarity against the locked
reference, a drift curve, and the shot where it starts to fail.

Three modes:

``images`` (default)
    Takes an ``IMAGE`` batch of generated frames plus a locked reference
    (``embedding`` socket, ``reference_image``, or ``element.embed_path``) and
    embeds every frame itself with SCRFD + ArcFace. This is the path that
    makes the claim true: the node sees pixels.

``embeddings``
    Takes pre-computed vectors as JSON. Kept for offline scoring and for
    comparing against an external embedder.

``scores``
    Takes finished per-shot scores. Kept for replaying a report.

The honest framing is the product. A measured partial failure is a stronger
result than an unmeasured claim, and it is what makes this node worth
demoing.
"""

from __future__ import annotations

import json

from .core.drift import DRIFT_FLOOR, DriftReport, build_report
from .core.embed import (
    EmbedderUnavailable,
    embed_image,
    load_embedding,
    to_rgb_batch,
)
from .core.types import DRIFT_REPORT, ELEMENT, EMBEDDING

CATEGORY = "SoulFoundry/evidence"

MODES = ("images", "embeddings", "scores")


class ConsistencyAudit:
    """Score generated frames against a locked reference."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "mode": (list(MODES), {"default": "images"}),
            },
            "optional": {
                "frames": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "baseline": ("IMAGE",),
                "embedding": (EMBEDDING,),
                "element": (ELEMENT,),
                "reference_json": ("STRING", {"default": "[]", "multiline": True}),
                "frames_json": ("STRING", {"default": "[]", "multiline": True}),
                "scores_json": ("STRING", {"default": "[]", "multiline": True}),
                "baseline_json": ("STRING", {"default": "[]", "multiline": True}),
                "seqs_json": ("STRING", {"default": "[]", "multiline": True}),
                "label": ("STRING", {"default": "locked", "multiline": False}),
                "project_path": ("STRING", {"default": "", "multiline": False}),
                "backend": (["auto", "arcface", "perceptual"], {"default": "auto"}),
                "allow_perceptual": ("BOOLEAN", {"default": False}),
                "model_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (DRIFT_REPORT, "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("report", "mean_similarity", "holding", "markdown")
    FUNCTION = "audit"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Per-shot face-similarity scoring against a locked Element, with a "
        "drift curve, trend slope, and the first shot that drops below the "
        "floor. mode='images' embeds real frames with ArcFace; "
        "mode='embeddings' and mode='scores' take pre-computed values."
    )

    def audit(self, mode: str = "images", frames=None, reference_image=None,
              baseline=None, embedding: dict | None = None,
              element: dict | None = None, reference_json: str = "[]",
              frames_json: str = "[]", scores_json: str = "[]",
              baseline_json: str = "[]", seqs_json: str = "[]",
              label: str = "locked", project_path: str = "",
              backend: str = "auto", allow_perceptual: bool = False,
              model_dir: str = "") -> tuple[dict, float, bool, str]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

        seqs = _load_list(seqs_json, "seqs_json", cast=int)
        baseline_scores = _load_list(baseline_json, "baseline_json") or None

        if mode == "scores":
            report = self._from_scores(scores_json, seqs, baseline_scores, label)

        elif mode == "embeddings":
            report = self._from_embeddings(
                reference_json, frames_json, baseline_scores, seqs, label
            )

        else:  # images
            report = self._from_images(
                frames=frames,
                reference_image=reference_image,
                baseline=baseline,
                embedding=embedding,
                element=element,
                reference_json=reference_json,
                baseline_json=baseline_json,
                seqs=seqs,
                label=label,
                backend=backend,
                allow_perceptual=allow_perceptual,
                model_dir=model_dir,
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

    # -- modes -------------------------------------------------------------

    def _from_scores(self, scores_json: str, seqs, baseline, label: str) -> DriftReport:
        scores = _load_list(scores_json, "scores_json", cast=float)
        if not scores:
            raise ValueError("mode='scores' needs a non-empty scores_json")
        return DriftReport(
            scores=scores,
            seqs=seqs or list(range(1, len(scores) + 1)),
            baseline=baseline,
            label=label,
        )

    def _from_embeddings(self, reference_json: str, frames_json: str,
                         baseline, seqs, label: str) -> DriftReport:
        reference = _load_list(reference_json, "reference_json", cast=float)
        if not reference:
            raise ValueError("mode='embeddings' needs a reference_json vector")
        frames = _load_vectors(frames_json, "frames_json")
        if not frames:
            raise ValueError("mode='embeddings' needs a non-empty frames_json")
        return build_report(
            reference, frames, seqs=seqs or None, baseline_frames=baseline, label=label
        )

    def _from_images(self, *, frames, reference_image, baseline, embedding,
                     element, reference_json, baseline_json, seqs, label,
                     backend, allow_perceptual, model_dir) -> DriftReport:
        if frames is None:
            raise ValueError(
                "mode='images' needs an IMAGE batch on the frames input. Use "
                "mode='embeddings' to score vectors you already have."
            )

        notes: list[str] = []
        reference, ref_backend = self._reference_vector(
            reference_image=reference_image,
            embedding=embedding,
            element=element,
            reference_json=reference_json,
            backend=backend,
            allow_perceptual=allow_perceptual,
            model_dir=model_dir,
        )
        if ref_backend != "arcface":
            notes.append(
                f"reference backend is {ref_backend}: these are NOT face-identity scores"
            )

        frame_vectors, missing = self._embed_frames(
            frames,
            backend=backend,
            allow_perceptual=allow_perceptual,
            model_dir=model_dir,
        )
        if missing:
            notes.append(
                f"{missing} frame(s) had no detectable face and scored 0.0; "
                f"a missing face is maximal drift, not a skipped shot"
            )
        if not frame_vectors:
            raise ValueError(
                "no faces were detected in any frame. The audit needs at least "
                "one frame the detector can see."
            )

        report = build_report(reference, frame_vectors, seqs=seqs or None, label=label)

        if baseline is not None:
            base_vectors, base_missing = self._embed_frames(
                baseline,
                backend=backend,
                allow_perceptual=allow_perceptual,
                model_dir=model_dir,
            )
            if base_vectors:
                from .core.drift import score_against_reference

                report.baseline = score_against_reference(reference, base_vectors)
            if base_missing:
                notes.append(f"{base_missing} baseline frame(s) had no detectable face")

        report.backend = ref_backend
        report.notes = notes
        return report

    # -- helpers -----------------------------------------------------------

    def _reference_vector(self, *, reference_image, embedding, element,
                          reference_json, backend, allow_perceptual,
                          model_dir) -> tuple[list[float], str]:
        # 1. an EMBEDDING socket from CharacterEmbed / EmbeddingFromFile
        if embedding and embedding.get("vector"):
            used = embedding.get("backend", "unknown")
            return list(embedding["vector"]), used

        # 2. a locked Element with a stored vector
        if element and element.get("embed_path"):
            vector, meta = load_embedding(element["embed_path"])
            return vector, meta.get("backend", "unknown")

        # 3. a reference image to embed now
        if reference_image is not None:
            try:
                result = embed_image(
                    reference_image,
                    backend=backend,
                    model_dir=model_dir or None,
                    allow_perceptual=allow_perceptual,
                    max_faces=1,
                )
            except EmbedderUnavailable as exc:
                raise ValueError(
                    f"no face embedder available: {exc}. Run "
                    f"scripts/fetch_models.py, or set allow_perceptual=True."
                ) from exc
            if not result.ok:
                detail = "; ".join(result.notes) or "no face detected"
                raise ValueError(f"no face found in the reference image ({detail})")
            return result.primary, result.backend

        # 4. a raw vector as JSON
        reference = _load_list(reference_json, "reference_json", cast=float)
        if reference:
            return reference, "external"

        raise ValueError(
            "mode='images' needs a reference: connect an embedding socket, an "
            "element with a stored embed_path, a reference_image, or pass "
            "reference_json."
        )

    def _embed_frames(self, frames, *, backend, allow_perceptual,
                      model_dir) -> tuple[list[list[float]], int]:
        """Embed every frame. Returns (vectors, frames_with_no_face)."""
        from .core.embed import get_embedder

        images = to_rgb_batch(frames)
        if not images:
            raise ValueError("frames input is an empty batch")

        missing = 0
        vectors: list[list[float]] = []

        if backend == "perceptual":
            if not allow_perceptual:
                raise ValueError(
                    "backend='perceptual' is not a face embedding; pass "
                    "allow_perceptual=True to opt in explicitly"
                )
            from .core.embed import perceptual_descriptor

            for img in images:
                vectors.append(perceptual_descriptor(img))
            return vectors, 0

        embedder = get_embedder(backend, model_dir)
        assert embedder is not None

        for img in images:
            result = embedder.embed(img, max_faces=1)
            if result.ok:
                vectors.append(result.primary)
            else:
                missing += 1
                # a frame with no face is maximally drifted; 0.0 keeps the
                # shot in the curve instead of silently dropping it
                vectors.append([0.0] * (result.dim or 512))

        if missing and not allow_perceptual:
            # keep the zeros: they are the honest reading, and the note says so
            pass
        return vectors, missing


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
