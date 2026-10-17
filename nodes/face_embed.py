"""CharacterEmbed: turn a reference image into an embedding.

This is the sensor. ``CharacterLock`` records *how* a character was locked;
this node produces the thing that gets compared later, and without it
``ConsistencyAudit`` was scoring vectors a user had to conjure from nowhere.

Backend policy is strict on purpose. ``auto`` resolves to ArcFace or raises;
it never silently degrades to a descriptor that is not a face embedding.
Passing ``backend="perceptual"`` (or ``allow_perceptual=True``) is an explicit
opt-in and every downstream report is labelled accordingly.
"""

from __future__ import annotations

from .core.embed import (
    BACKENDS,
    EMBED_DIM,
    EmbedderUnavailable,
    embed_image,
    save_embedding,
)
from .core.types import EMBEDDING

CATEGORY = "SoulFoundry/lock"


class CharacterEmbed:
    """Embed a reference face, optionally persisting it next to the project."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "backend": (list(BACKENDS), {"default": "auto"}),
                "allow_perceptual": ("BOOLEAN", {"default": False}),
                "save_path": ("STRING", {"default": "", "multiline": False}),
                "max_faces": ("INT", {"default": 1, "min": 0, "max": 64}),
                "model_dir": ("STRING", {"default": "", "multiline": False}),
                "label": ("STRING", {"default": "reference", "multiline": False}),
            },
        }

    RETURN_TYPES = (EMBEDDING, "STRING", "STRING", "INT")
    RETURN_NAMES = ("embedding", "vector_json", "status", "faces_found")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Detect and embed faces from an IMAGE using SCRFD + ArcFace on CPU. "
        "Emits a normalised 512-d vector for the largest face and can write it "
        "to disk as .npy plus a JSON sidecar. Never silently falls back to a "
        "non-identity descriptor."
    )

    def run(self, image, backend: str = "auto", allow_perceptual: bool = False,
            save_path: str = "", max_faces: int = 1,
            model_dir: str = "", label: str = "reference"
            ) -> tuple[dict, str, str, int]:
        try:
            result = embed_image(
                image,
                backend=backend,
                model_dir=model_dir or None,
                allow_perceptual=allow_perceptual,
                max_faces=max_faces,
            )
        except EmbedderUnavailable as exc:
            raise ValueError(
                f"no face embedder available: {exc}. Run "
                f"scripts/fetch_models.py to install the ArcFace models, or set "
                f"allow_perceptual=True to accept a non-identity descriptor."
            ) from exc

        if not result.ok:
            detail = "; ".join(result.notes) or "no face detected"
            raise ValueError(
                f"no face found in the reference image ({detail}). A character "
                f"lock needs a face the detector can see."
            )

        primary = result.primary
        faces = len(result.vectors)

        path = ""
        if save_path:
            written = save_embedding(
                primary,
                save_path,
                meta={
                    "backend": result.backend,
                    "is_face_embedding": result.backend == "arcface",
                    "dim": result.dim,
                    "label": label,
                    "faces_in_image": faces,
                },
            )
            path = str(written)

        import json

        payload = {
            "vector": primary,
            "dim": result.dim,
            "backend": result.backend,
            "is_face_embedding": result.backend == "arcface",
            "path": path,
            "label": label,
            "faces": [f.to_dict() for f in result.faces],
            "all_vectors": result.vectors,
            "notes": result.notes,
        }

        if result.backend == "arcface":
            status = (
                f"embedded @{label}: {faces} face(s), {result.dim}-d ArcFace"
            )
        else:
            status = (
                f"embedded @{label}: PERCEPTUAL DESCRIPTOR, not face identity "
                f"({faces} vector(s), {result.dim}-d). Do not publish as a face score."
            )
        if path:
            status += f" -> {path}"

        return (payload, json.dumps(primary), status, faces)


class EmbeddingFromFile:
    """Load a previously saved embedding. Keeps audits reproducible."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (EMBEDDING, "STRING", "STRING")
    RETURN_NAMES = ("embedding", "vector_json", "status")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Load a .npy embedding and its JSON sidecar written by CharacterEmbed. "
        "The sidecar records which backend produced it, so a perceptual "
        "descriptor cannot be mistaken for an ArcFace vector later."
    )

    def run(self, path: str) -> tuple[dict, str, str]:
        from .core.embed import load_embedding

        if not path:
            raise ValueError("path is empty")
        vector, meta = load_embedding(path)
        backend = meta.get("backend", "unknown")
        payload = {
            "vector": vector,
            "dim": len(vector),
            "backend": backend,
            "is_face_embedding": backend == "arcface",
            "path": path,
            "label": meta.get("label", "reference"),
            "faces": [],
            "all_vectors": [vector],
            "notes": meta.get("warning", "") and [meta["warning"]] or [],
        }
        return (payload, __import__("json").dumps(vector), f"loaded {len(vector)}-d {backend} from {path}")


__all__ = ["CharacterEmbed", "EmbeddingFromFile", "EMBED_DIM"]
