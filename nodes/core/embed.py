"""Face embedding: the sensor the drift metric was missing.

Before this module, ``ConsistencyAudit`` could only score vectors somebody
else produced. That made the headline feature a calculator with no sensor.
This module is the sensor.

Backends, in order of preference:

``arcface``
    SCRFD face detection + ArcFace (MobileFaceNet, 512-d) embedding, run
    through onnxruntime on CPU. No torch, no GPU, no insightface package.
    This is the real path and the default.

``perceptual``
    A deterministic image descriptor (block-mean luminance + colour
    histogram). It correlates with visual similarity but it is **not** face
    identity, and it is **never** used unless explicitly allowed. It exists
    for CI and for machines with no ONNX runtime, and every result it
    produces is labelled so no report can silently pass it off as a face
    match.

The default is deliberately strict: ``backend="auto"`` raises rather than
quietly degrading to ``perceptual``. A number that looks like a face score
but is not one is worse than an error, because it ends up in a demo video.

Pure numpy + onnxruntime. No torch, no cv2, no insightface, no skimage: the
SCRFD post-processing, the 5-point similarity alignment and the bilinear warp
are all implemented here so the dependency footprint stays small enough for
ComfyUI's environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

#: ArcFace's canonical 112x112 five-point template (left eye, right eye, nose,
#: left mouth corner, right mouth corner). Every ArcFace checkpoint is trained
#: against this alignment, so the crop has to match it or the embedding is
#: measuring the crop, not the face.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

EMBED_SIZE = 112
EMBED_DIM = 512

#: SCRFD strides. det_500m emits three feature levels at 8/16/32.
_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
_DET_SIZE = 640

DET_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

BACKENDS = ("auto", "arcface", "perceptual")


class EmbedderUnavailable(RuntimeError):
    """No usable face-embedding backend.

    Raised instead of silently falling back to a descriptor that is not a
    face embedding, because a plausible-looking number is more damaging than
    a clear failure.
    """


# --------------------------------------------------------------------------
# image normalisation
# --------------------------------------------------------------------------


def to_rgb_uint8(image: Any) -> np.ndarray:
    """Normalise anything ComfyUI or a test might hand us into HWC uint8 RGB.

    Accepts, in the order ComfyUI actually produces them:

    * a torch tensor ``[B, H, W, C]`` or ``[H, W, C]``, float in ``0..1``
      (this is what an ``IMAGE`` socket carries, and it is the case the old
      ``qc_sheet`` crashed on: tensors have no ``.convert()``)
    * a numpy array, float ``0..1`` or ``0..255``, HWC or CHW
    * a PIL ``Image``
    * ``bytes`` (encoded file) or ``str``/``os.PathLike`` (a path)
    """
    # torch tensor: duck-typed so torch is not a hard import
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        arr = image.detach().cpu().float().numpy()
        return _array_to_rgb(arr)

    if isinstance(image, (bytes, bytearray)):
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(bytes(image))) as im:
            return _pil_to_rgb(im)

    if isinstance(image, (str, os.PathLike)):
        from PIL import Image

        with Image.open(os.fspath(image)) as im:
            return _pil_to_rgb(im)

    if isinstance(image, np.ndarray):
        return _array_to_rgb(image)

    # PIL last, since a PIL image is neither of the above
    if hasattr(image, "convert") and hasattr(image, "size"):
        return _pil_to_rgb(image)

    raise TypeError(f"cannot normalise {type(image)!r} into an RGB image")


def to_rgb_batch(image: Any) -> list[np.ndarray]:
    """Split a ComfyUI ``IMAGE`` batch into per-frame HWC uint8 arrays.

    An ``IMAGE`` socket carries ``[B, H, W, C]``. A drift audit needs every
    frame in the batch scored, not just the first, so this exists separately
    from :func:`to_rgb_uint8` (which takes element 0 by design, because a
    reference is one face).
    """
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        arr = image.detach().cpu().float().numpy()
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        return [to_rgb_uint8(image)]

    if arr.ndim == 4:
        if arr.shape[0] == 0:
            return []
        return [_array_to_rgb(arr[i]) for i in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        return [_array_to_rgb(arr)]
    if arr.ndim == 3:
        return [_array_to_rgb(arr)]
    return [to_rgb_uint8(image)]


def _pil_to_rgb(im: Any) -> np.ndarray:
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        canvas = _flatten_alpha(im)
    else:
        canvas = im.convert("RGB")
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def _flatten_alpha(im: Any) -> Any:
    """Composite a transparent image onto mid-grey instead of black.

    Face detection on a black background loses the jawline on dark hair, and
    a transparent PNG is a common input for a character sheet.
    """
    from PIL import Image

    rgba = im.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (128, 128, 128))
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg


def _array_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        # [B, H, W, C] -> first frame. Batched audits call this per frame.
        if arr.shape[0] == 0:
            raise ValueError("empty batch: IMAGE tensor has zero frames")
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected an image array, got shape {arr.shape}")

    channels = arr.shape[-1]
    if channels == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif channels == 4:
        # composite onto mid-grey; detection on a black matte loses the
        # jawline against dark hair
        alpha = arr[..., 3:4].astype(np.float32)
        rgb = arr[..., :3].astype(np.float32)
        if alpha.size and float(alpha.max()) <= 1.0001:
            alpha = alpha * 255.0
        if rgb.size and float(rgb.max()) <= 1.0001:
            rgb = rgb * 255.0
        a = np.clip(alpha, 0, 255) / 255.0
        arr = rgb * a + 128.0 * (1.0 - a)
        return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif channels != 3:
        raise ValueError(f"expected 1, 3 or 4 channels, got {channels}")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    arr = arr.astype(np.float32)
    # float images from ComfyUI are 0..1; anything peaking above 1 is 0..255
    if arr.size and float(arr.max()) <= 1.0001:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------
# geometry: similarity transform + bilinear warp (no cv2 / skimage)
# --------------------------------------------------------------------------


def estimate_similarity(
    src: np.ndarray, dst: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares 2D similarity (scale, rotation, translation) src -> dst.

    This is Umeyama's method, which is what ``cv2.estimateAffinePartial2D``
    and ``skimage.transform.SimilarityTransform`` both reduce to. Implemented
    here so alignment needs neither.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_d = src - src_mean
    dst_d = dst - dst_mean

    cov = (dst_d.T @ src_d) / n
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    d_matrix = np.diag([1.0, d])
    rotation = u @ d_matrix @ vt

    var_src = float((src_d**2).sum() / n)
    if var_src <= 1e-12:
        raise ValueError("degenerate landmark set: all points coincide")
    scale = float((s * np.diag(d_matrix)).sum() / var_src)
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def warp_similarity(
    image: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    out_size: int = EMBED_SIZE,
) -> np.ndarray:
    """Bilinearly sample ``image`` onto an ``out_size`` square.

    ``scale``/``rotation``/``translation`` map *destination* -> *source*, so
    this is a pull (gather) warp and no output pixel is left unfilled.
    """
    ys, xs = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
    dst = np.stack([xs.ravel(), ys.ravel()], axis=-1)  # (N, 2)

    # inverse of: dst = scale * R @ src + t   =>   src = R^T (dst - t) / scale
    src_xy = ((dst - translation) @ rotation) / scale
    sx = src_xy[:, 0].reshape(out_size, out_size)
    sy = src_xy[:, 1].reshape(out_size, out_size)

    h, w = image.shape[:2]
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)

    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (sx - x0)[..., None]
    wy = (sy - y0)[..., None]

    src_f = image.astype(np.float32)
    top = src_f[y0, x0] * (1 - wx) + src_f[y0, x1] * wx
    bottom = src_f[y1, x0] * (1 - wx) + src_f[y1, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def align_face(
    image: np.ndarray, landmarks: np.ndarray, out_size: int = EMBED_SIZE
) -> np.ndarray:
    """Crop and align a face to ArcFace's canonical 112x112 frame."""
    template = ARCFACE_TEMPLATE
    if out_size != EMBED_SIZE:
        template = template * (out_size / EMBED_SIZE)
    scale, rotation, translation = estimate_similarity(
        np.asarray(landmarks, dtype=np.float64), template.astype(np.float64)
    )
    # estimate_similarity maps landmarks -> template, i.e. dst -> src for the
    # warp, which is exactly the pull direction warp_similarity expects.
    return warp_similarity(image, scale, rotation, translation, out_size)


# --------------------------------------------------------------------------
# SCRFD detection
# --------------------------------------------------------------------------


@dataclass
class Face:
    """One detected face: box, landmarks, confidence, area."""

    box: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [round(v, 2) for v in self.box],
            "score": round(self.score, 4),
            "landmarks": [[round(float(a), 2), round(float(b), 2)] for a, b in self.landmarks],
        }


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        out.append(px)
        out.append(py)
    return np.stack(out, axis=-1)


def _nms(dets: np.ndarray, threshold: float) -> list[int]:
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = (dets[:, i] for i in range(5))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou <= threshold]
    return keep


# --------------------------------------------------------------------------
# embedder
# --------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """Embeddings for one image, plus how they were produced."""

    vectors: list[list[float]] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    backend: str = ""
    dim: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.vectors)

    @property
    def primary(self) -> list[float]:
        """The largest face, which is the subject in every framing we care about."""
        if not self.vectors:
            raise ValueError("no faces embedded")
        return self.vectors[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "faces": [f.to_dict() for f in self.faces],
            "vectors": self.vectors,
            "notes": self.notes,
            "is_face_embedding": self.backend == "arcface",
        }


class FaceEmbedder:
    """SCRFD detection + ArcFace embedding, onnxruntime CPU.

    Lazily loads the two ONNX sessions on first use so importing the node pack
    stays instant and a machine without the models still loads ComfyUI.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        det_threshold: float = DET_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self._det: Any = None
        self._rec: Any = None

    # -- models ------------------------------------------------------------

    @property
    def det_path(self) -> Path:
        return self.model_dir / "det_500m.onnx"

    @property
    def rec_path(self) -> Path:
        return self.model_dir / "w600k_mbf.onnx"

    def available(self) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "onnxruntime is not installed"
        if not self.det_path.is_file():
            return False, f"missing detector: {self.det_path}"
        if not self.rec_path.is_file():
            return False, f"missing recogniser: {self.rec_path}"
        return True, "ok"

    def _sessions(self) -> tuple[Any, Any]:
        if self._det is None or self._rec is None:
            ok, why = self.available()
            if not ok:
                raise EmbedderUnavailable(why)
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._det = ort.InferenceSession(
                str(self.det_path), opts, providers=["CPUExecutionProvider"]
            )
            self._rec = ort.InferenceSession(
                str(self.rec_path), opts, providers=["CPUExecutionProvider"]
            )
        return self._det, self._rec

    # -- detect ------------------------------------------------------------

    def detect(self, rgb: np.ndarray) -> list[Face]:
        """SCRFD forward pass. Returns faces sorted largest first."""
        det, _ = self._sessions()
        img = to_rgb_uint8(rgb)

        h, w = img.shape[:2]
        scale = min(_DET_SIZE / w, _DET_SIZE / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = _resize_bilinear(img, new_h, new_w)

        blob = np.zeros((_DET_SIZE, _DET_SIZE, 3), dtype=np.float32)
        blob[:new_h, :new_w] = resized.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        input_name = det.get_inputs()[0].name
        outs = det.run(None, {input_name: np.ascontiguousarray(blob)})

        fmc = 3
        scores_all, boxes_all, kps_all = [], [], []
        for idx, stride in enumerate(_STRIDES):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + fmc].reshape(-1, 4) * stride
            kps_preds = outs[idx + fmc * 2].reshape(-1, 10) * stride

            feat_h = _DET_SIZE // stride
            feat_w = _DET_SIZE // stride
            anchor_centers = np.stack(
                np.mgrid[:feat_h, :feat_w][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            anchor_centers = np.stack([anchor_centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)

            pos = np.where(scores >= self.det_threshold)[0]
            if pos.size == 0:
                continue
            scores_all.append(scores[pos])
            boxes_all.append(_distance2bbox(anchor_centers[pos], bbox_preds[pos]))
            kps_all.append(_distance2kps(anchor_centers[pos], kps_preds[pos]))

        if not scores_all:
            return []

        scores = np.concatenate(scores_all)
        boxes = np.concatenate(boxes_all)
        kps = np.concatenate(kps_all)

        # map back out of the letterboxed 640 canvas
        boxes = boxes / scale
        kps = kps / scale

        dets = np.hstack([boxes, scores[:, None]])
        keep = _nms(dets, self.nms_threshold)

        faces = []
        for i in keep:
            faces.append(
                Face(
                    box=tuple(float(v) for v in boxes[i]),
                    score=float(scores[i]),
                    landmarks=kps[i].reshape(5, 2).astype(np.float32),
                )
            )
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    # -- embed -------------------------------------------------------------

    def embed(self, image: Any, max_faces: int = 0) -> EmbeddingResult:
        """Detect and embed every face in ``image``.

        ``max_faces=0`` means no cap. Vectors are L2-normalised, so cosine
        similarity is a dot product.
        """
        rgb = to_rgb_uint8(image)
        faces = self.detect(rgb)
        if max_faces:
            faces = faces[:max_faces]

        _, rec = self._sessions()
        input_name = rec.get_inputs()[0].name

        vectors: list[list[float]] = []
        notes: list[str] = []
        for face in faces:
            crop = align_face(rgb, face.landmarks)
            blob = (crop - 127.5) / 127.5
            blob = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)
            out = rec.run(None, {input_name: np.ascontiguousarray(blob)})[0]
            vec = np.asarray(out, dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                notes.append("zero embedding returned for a detected face")
                continue
            vectors.append((vec / norm).tolist())

        if not faces:
            notes.append("no face detected")
        return EmbeddingResult(
            vectors=vectors,
            faces=faces,
            backend="arcface",
            dim=EMBED_DIM,
            notes=notes,
        )


def _resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Bilinear resize, implemented so detection needs neither cv2 nor PIL."""
    if img.shape[0] == new_h and img.shape[1] == new_w:
        return img
    h, w = img.shape[:2]
    ys = (np.arange(new_h, dtype=np.float32) + 0.5) * (h / new_h) - 0.5
    xs = (np.arange(new_w, dtype=np.float32) + 0.5) * (w / new_w) - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    src = img.astype(np.float32)
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bottom * wy, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# perceptual fallback (explicitly NOT face identity)
# --------------------------------------------------------------------------


def perceptual_descriptor(image: Any, size: int = 16) -> list[float]:
    """A deterministic visual descriptor. Not a face embedding.

    Block-mean luminance plus a coarse RGB histogram, concatenated and
    L2-normalised. Two photos of the same scene land close together; it has
    no notion of identity. Callers must opt in via ``allow_perceptual=True``
    and every consumer labels it ``backend="perceptual"``.
    """
    rgb = to_rgb_uint8(image).astype(np.float32)
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty image")

    # block means over a size x size grid
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    lum = rgb.mean(axis=2)
    blocks = np.empty((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            patch = lum[ys[i] : max(ys[i] + 1, ys[i + 1]), xs[j] : max(xs[j] + 1, xs[j + 1])]
            blocks[i, j] = patch.mean() if patch.size else 0.0
    blocks = (blocks - blocks.mean()) / (blocks.std() + 1e-6)

    # coarse colour histogram, 8 bins per channel
    hist_parts = []
    for c in range(3):
        hist, _ = np.histogram(rgb[..., c], bins=8, range=(0, 255))
        hist = hist.astype(np.float32)
        hist_parts.append(hist / (hist.sum() + 1e-6))
    hist_vec = np.concatenate(hist_parts)

    vec = np.concatenate([blocks.ravel(), hist_vec * 4.0]).astype(np.float64)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist() if norm > 1e-12 else vec.tolist()


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------


def default_model_dir() -> Path:
    """Where the ONNX models live.

    ``SOULFOUNDRY_MODEL_DIR`` wins, then the repo's ``models/buffalo_sc``,
    then a per-user cache so a pip install has somewhere to put them.
    """
    env = os.environ.get("SOULFOUNDRY_MODEL_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2] / "models" / "buffalo_sc"
    if repo.is_dir():
        return repo
    return Path.home() / ".cache" / "soulfoundry" / "buffalo_sc"


def get_embedder(
    backend: str = "auto",
    model_dir: str | os.PathLike[str] | None = None,
) -> FaceEmbedder | None:
    """Resolve a backend. Returns None only for ``perceptual``."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "perceptual":
        return None

    embedder = FaceEmbedder(model_dir=model_dir)
    ok, why = embedder.available()
    if ok:
        return embedder
    if backend == "arcface":
        raise EmbedderUnavailable(why)
    # auto: report precisely why the real path is unavailable
    raise EmbedderUnavailable(
        f"no face-embedding backend available ({why}). Install onnxruntime and "
        f"place det_500m.onnx + w600k_mbf.onnx in {embedder.model_dir}, set "
        f"SOULFOUNDRY_MODEL_DIR, or pass allow_perceptual=True to accept a "
        f"non-identity descriptor."
    )


def embed_image(
    image: Any,
    backend: str = "auto",
    model_dir: str | os.PathLike[str] | None = None,
    allow_perceptual: bool = False,
    max_faces: int = 0,
) -> EmbeddingResult:
    """Embed one image, honouring the strict-by-default policy."""
    if backend == "perceptual":
        if not allow_perceptual:
            raise EmbedderUnavailable(
                "backend='perceptual' is not a face embedding; pass "
                "allow_perceptual=True to opt in explicitly"
            )
        return EmbeddingResult(
            vectors=[perceptual_descriptor(image)],
            faces=[],
            backend="perceptual",
            dim=len(perceptual_descriptor(image)),
            notes=[
                "perceptual descriptor: NOT face identity, do not present as a face score"
            ],
        )

    embedder = get_embedder(backend, model_dir)
    assert embedder is not None
    result = embedder.embed(image, max_faces=max_faces)
    if not result.ok and allow_perceptual:
        desc = perceptual_descriptor(image)
        return EmbeddingResult(
            vectors=[desc],
            faces=result.faces,
            backend="perceptual",
            dim=len(desc),
            notes=result.notes
            + [
                "fell back to perceptual descriptor: NOT face identity, "
                "do not present as a face score"
            ],
        )
    return result


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save_embedding(
    vector: Sequence[float], path: str | os.PathLike[str], meta: dict[str, Any] | None = None
) -> Path:
    """Write a vector as ``.npy`` plus a JSON sidecar.

    The sidecar matters: an ``.npy`` on its own does not record which backend
    produced it, and a perceptual descriptor must never be mistaken for an
    ArcFace vector months later.
    """
    target = Path(path)
    if target.suffix != ".npy":
        target = target.with_suffix(".npy")
    target.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(list(vector), dtype=np.float32)
    np.save(target, arr)

    payload = {
        "path": target.name,
        "dim": int(arr.shape[0]),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
    payload.update(meta or {})
    target.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def load_embedding(path: str | os.PathLike[str]) -> tuple[list[float], dict[str, Any]]:
    """Read a vector and its sidecar. Sidecar is optional but never ignored."""
    target = Path(path)
    if target.suffix != ".npy":
        candidate = target.with_suffix(".npy")
        if candidate.is_file():
            target = candidate
    if not target.is_file():
        raise FileNotFoundError(f"no embedding at {target}")
    arr = np.load(target)
    vec = [float(v) for v in np.asarray(arr).reshape(-1)]
    side = target.with_suffix(".json")
    meta: dict[str, Any] = {}
    if side.is_file():
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {"warning": "unreadable sidecar"}
    return vec, meta


__all__ = [
    "ARCFACE_TEMPLATE",
    "BACKENDS",
    "EMBED_DIM",
    "EmbedderUnavailable",
    "EmbeddingResult",
    "Face",
    "FaceEmbedder",
    "align_face",
    "default_model_dir",
    "embed_image",
    "estimate_similarity",
    "get_embedder",
    "load_embedding",
    "perceptual_descriptor",
    "save_embedding",
    "to_rgb_batch",
    "to_rgb_uint8",
    "warp_similarity",
]
