"""Tests for the face embedder and the image-mode audit.

These are the tests that would have caught the two defects the audit found:

* ``qc_sheet`` crashing on a real ComfyUI ``IMAGE`` (a torch tensor), because
  nothing exercised the image path;
* ``ConsistencyAudit`` and ``CharacterLock`` never touching pixels, because
  nothing asserted a vector came out of an image.

The ArcFace tests run against real public-domain portraits in
``tests/fixtures/faces`` and skip cleanly when the ONNX models are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nodes.core.embed import (
    ARCFACE_TEMPLATE,
    EMBED_DIM,
    EmbedderUnavailable,
    FaceEmbedder,
    align_face,
    default_model_dir,
    embed_image,
    estimate_similarity,
    load_embedding,
    perceptual_descriptor,
    save_embedding,
    to_rgb_batch,
    to_rgb_uint8,
)
from nodes.core.drift import cosine_similarity

FACES = Path(__file__).parent / "fixtures" / "faces"
LINCOLN_A = FACES / "lincoln_a.jpg"
LINCOLN_B = FACES / "lincoln_b.jpg"
LINCOLN_C = FACES / "lincoln_c.jpg"
DARWIN_A = FACES / "darwin_a.jpg"
DARWIN_B = FACES / "darwin_b.jpg"
CURIE_A = FACES / "curie_a.jpg"


def _have_models() -> bool:
    return FaceEmbedder().available()[0]


def _have_faces() -> bool:
    return all(p.is_file() for p in (LINCOLN_A, LINCOLN_B, DARWIN_A, CURIE_A))


needs_models = pytest.mark.skipif(
    not _have_models(), reason="ArcFace ONNX models not installed"
)
needs_faces = pytest.mark.skipif(
    not _have_faces(), reason="face fixtures not downloaded"
)


# --------------------------------------------------------------------------
# the crash the audit found: torch tensors have no .convert()
# --------------------------------------------------------------------------


class FakeTensor:
    """Duck-typed stand-in for a torch tensor.

    ComfyUI hands ``IMAGE`` sockets a ``[B, H, W, C]`` float tensor in 0..1.
    The old ``qc_sheet`` called ``.convert("L")`` on it and raised
    ``AttributeError``. This stub has exactly the attributes the real
    normaliser probes (``detach``/``cpu``/``float``/``numpy``) and nothing
    else, so if normalisation ever regresses to assuming PIL, this fails.
    """

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._array


def test_to_rgb_uint8_accepts_a_batched_float_tensor():
    arr = np.zeros((2, 8, 8, 3), dtype=np.float32)
    arr[..., 0] = 1.0
    out = to_rgb_uint8(FakeTensor(arr))
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 255


def test_to_rgb_uint8_accepts_chw_and_grayscale():
    chw = np.full((3, 4, 5), 0.5, dtype=np.float32)
    out = to_rgb_uint8(FakeTensor(chw[None, ...]))
    assert out.shape == (4, 5, 3)

    grey = np.full((4, 5, 1), 200, dtype=np.uint8)
    out = to_rgb_uint8(grey)
    assert out.shape == (4, 5, 3)
    assert out[0, 0, 1] == 200


def test_to_rgb_batch_returns_every_frame():
    arr = np.zeros((3, 6, 7, 3), dtype=np.float32)
    frames = to_rgb_batch(FakeTensor(arr))
    assert len(frames) == 3
    assert all(f.shape == (6, 7, 3) for f in frames)


def test_qc_sheet_accepts_a_torch_like_tensor():
    """The exact call that used to raise AttributeError."""
    from nodes.core.sheet import qc_sheet

    # a plausible 3-panel sheet: mid-grey backdrop, brighter subject column
    arr = np.full((1, 96, 192, 3), 0.5, dtype=np.float32)
    arr[:, :, 20:50] = 0.8
    arr[:, :, 84:114] = 0.8
    arr[:, :, 148:178] = 0.8
    result = qc_sheet(FakeTensor(arr))
    assert result.checks
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.passed, bool)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_estimate_similarity_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(5, 2)) * 20 + 100
    theta = np.deg2rad(30.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale = 1.7
    shift = np.array([12.0, -8.0])
    dst = scale * (src @ rot.T) + shift

    got_scale, got_rot, got_shift = estimate_similarity(src, dst)
    assert got_scale == pytest.approx(scale, rel=1e-6)
    assert np.allclose(got_rot, rot, atol=1e-6)
    assert np.allclose(got_shift, shift, atol=1e-6)


def test_align_face_maps_the_template_onto_itself():
    """If the landmarks already equal the template, the crop must be identity."""
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    for i, (x, y) in enumerate(ARCFACE_TEMPLATE):
        img[int(round(y)), int(round(x))] = (40 * (i + 1), 255, 0)
    out = align_face(img, ARCFACE_TEMPLATE.copy())
    assert out.shape == (112, 112, 3)
    # warping by the identity transform must preserve the sampled pixels
    for i, (x, y) in enumerate(ARCFACE_TEMPLATE):
        px = out[int(round(y)), int(round(x))]
        assert np.allclose(px, img[int(round(y)), int(round(x))], atol=12)


def test_estimate_similarity_rejects_degenerate_points():
    pts = np.ones((5, 2))
    with pytest.raises(ValueError, match="degenerate"):
        estimate_similarity(pts, ARCFACE_TEMPLATE)


# --------------------------------------------------------------------------
# strict backend policy
# --------------------------------------------------------------------------


def test_perceptual_is_opt_in_only():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    with pytest.raises(EmbedderUnavailable, match="not a face embedding"):
        embed_image(img, backend="perceptual", allow_perceptual=False)

    result = embed_image(img, backend="perceptual", allow_perceptual=True)
    assert result.backend == "perceptual"
    assert result.ok


def test_perceptual_descriptor_is_deterministic_and_unit_length():
    img = np.full((40, 40, 3), 90, dtype=np.uint8)
    img[:20] = 200
    a = perceptual_descriptor(img)
    b = perceptual_descriptor(img)
    assert a == b
    assert abs(sum(v * v for v in a) ** 0.5 - 1.0) < 1e-6


def test_auto_never_silently_degrades(monkeypatch, tmp_path):
    """auto must raise, not fall back, when no real model is present."""
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    empty = tmp_path / "no-models"
    empty.mkdir()
    with pytest.raises(EmbedderUnavailable):
        embed_image(img, backend="auto", model_dir=empty)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_save_and_load_roundtrip_with_sidecar(tmp_path):
    vec = [0.1, 0.2, 0.3]
    written = save_embedding(vec, tmp_path / "x", meta={"backend": "arcface"})
    assert written.suffix == ".npy"
    assert written.with_suffix(".json").is_file()

    loaded, meta = load_embedding(written)
    assert loaded == pytest.approx(vec, abs=1e-6)
    assert meta["backend"] == "arcface"
    assert meta["dim"] == 3


def test_load_missing_embedding_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_embedding(tmp_path / "nope.npy")


# --------------------------------------------------------------------------
# real ArcFace: the acceptance test
# --------------------------------------------------------------------------


@needs_models
@needs_faces
def test_arcface_separates_people():
    """Same person must score higher than different people.

    This is the claim the whole node pack rests on, so it is asserted with a
    real model on real photographs rather than on synthetic vectors.
    """
    embedder = FaceEmbedder()
    vecs = {}
    for name, path in [
        ("lincoln_a", LINCOLN_A),
        ("lincoln_b", LINCOLN_B),
        ("darwin_a", DARWIN_A),
        ("curie_a", CURIE_A),
    ]:
        result = embedder.embed(path)
        assert result.ok, f"no face found in {name}"
        assert result.dim == EMBED_DIM
        vecs[name] = result.primary

    same = cosine_similarity(vecs["lincoln_a"], vecs["lincoln_b"])
    cross = max(
        cosine_similarity(vecs["lincoln_a"], vecs["darwin_a"]),
        cosine_similarity(vecs["lincoln_a"], vecs["curie_a"]),
        cosine_similarity(vecs["darwin_a"], vecs["curie_a"]),
    )
    assert same > cross, f"same={same:.3f} cross={cross:.3f}"
    assert same > 0.5
    assert cross < 0.4


@needs_models
@needs_faces
def test_arcface_embeddings_are_unit_length():
    result = FaceEmbedder().embed(LINCOLN_A)
    norm = sum(v * v for v in result.primary) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5)


@needs_models
@needs_faces
def test_detector_returns_five_landmarks():
    result = FaceEmbedder().embed(LINCOLN_A)
    face = result.faces[0]
    assert face.landmarks.shape == (5, 2)
    assert face.score > 0.5
    x1, y1, x2, y2 = face.box
    assert x2 > x1 and y2 > y1


@needs_models
def test_embed_image_with_no_face_reports_cleanly():
    blank = np.full((128, 128, 3), 128, dtype=np.uint8)
    result = FaceEmbedder().embed(blank)
    assert not result.ok
    assert "no face detected" in result.notes


# --------------------------------------------------------------------------
# nodes wired to real pixels
# --------------------------------------------------------------------------


@needs_models
@needs_faces
def test_character_embed_node_returns_a_vector_and_saves_it(tmp_path):
    from nodes.face_embed import CharacterEmbed

    node = CharacterEmbed()
    payload, vector_json, status, faces = node.run(
        str(LINCOLN_A), save_path=str(tmp_path / "ref.npy")
    )
    assert faces == 1
    assert payload["dim"] == EMBED_DIM
    assert payload["backend"] == "arcface"
    assert payload["is_face_embedding"] is True
    assert len(payload["vector"]) == EMBED_DIM
    assert Path(payload["path"]).is_file()
    assert "ArcFace" in status


@needs_models
@needs_faces
def test_character_lock_computes_a_vector_from_an_image(tmp_path):
    from nodes.cast_param_grid import CastParamGrid
    from nodes.character_lock import CharacterLock

    spec = CastParamGrid().build(
        name="santiago", genre="Drama", budget=50, era="Contemporary",
        archetype="Everyman", gender="unspecified", race_ethnicity="unspecified",
        age="30s", build="Average", height="average height", eye_color="Brown",
        hair_style="Short", hair_texture="Straight", hair_color="Dark Brown",
        facial_hair="None", details="None", outfit="Casual",
    )[0]

    element, embedding, status = CharacterLock().lock(
        spec,
        route="embedding",
        image=str(LINCOLN_A),
        project_path=str(tmp_path / "proj"),
    )
    assert embedding["state"] == "computed"
    assert embedding["backend"] == "arcface"
    assert len(embedding["vector"]) == EMBED_DIM
    assert element["embed_path"]
    assert Path(element["embed_path"]).is_file()
    assert "computed" in status


def _uniform_batch(paths, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Build a uniform [B, H, W, 3] float batch, as ComfyUI would.

    Real ``IMAGE`` batches are always one resolution; the fixtures are not,
    so they get resized here rather than the test asserting on ragged input
    ComfyUI could never produce.
    """
    from PIL import Image

    frames = []
    for path in paths:
        with Image.open(path) as im:
            frames.append(np.asarray(im.convert("RGB").resize(size), dtype=np.uint8))
    return np.stack(frames).astype(np.float32) / 255.0


@needs_models
@needs_faces
def test_audit_scores_real_frames_end_to_end():
    """The whole claim, exercised: images in, drift report out."""
    from nodes.consistency_audit import ConsistencyAudit

    frames = FakeTensor(_uniform_batch([LINCOLN_B, LINCOLN_A, DARWIN_A]))

    report, mean, holding, markdown = ConsistencyAudit().audit(
        mode="images",
        frames=frames,
        reference_image=str(LINCOLN_A),
        label="locked",
    )
    assert report["shots"] == 3
    assert report["backend"] == "arcface"
    assert report["is_face_embedding"] is True
    assert report["curve"][0]["score"] > report["curve"][2]["score"]
    assert "ArcFace" in markdown


@needs_models
@needs_faces
def test_audit_scores_a_faceless_frame_as_zero():
    """A frame with no face is maximal drift, not a skipped shot."""
    from nodes.consistency_audit import ConsistencyAudit

    batch = _uniform_batch([LINCOLN_B, LINCOLN_B])
    batch[1] = 0.5  # flat grey: nothing to detect
    frames = FakeTensor(batch)

    report, mean, holding, markdown = ConsistencyAudit().audit(
        mode="images",
        frames=frames,
        reference_image=str(LINCOLN_A),
    )
    assert report["shots"] == 2
    assert report["curve"][1]["score"] == 0.0
    assert report["holding"] is False
    assert any("no detectable face" in note for note in report["notes"])


def test_audit_images_mode_requires_frames():
    from nodes.consistency_audit import ConsistencyAudit

    with pytest.raises(ValueError, match="needs an IMAGE batch"):
        ConsistencyAudit().audit(mode="images", reference_json="[1.0, 0.0]")


def test_audit_images_mode_requires_a_reference():
    from nodes.consistency_audit import ConsistencyAudit

    frames = FakeTensor(np.zeros((1, 8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="needs a reference"):
        ConsistencyAudit().audit(mode="images", frames=frames)


# --------------------------------------------------------------------------
# the report must never launder a perceptual descriptor as a face score
# --------------------------------------------------------------------------


def test_markdown_labels_a_non_face_backend():
    from nodes.core.drift import DriftReport

    report = DriftReport(scores=[0.9, 0.8], seqs=[1, 2], backend="perceptual")
    md = report.markdown()
    assert "Not a face-identity score" in md
    assert report.to_dict()["is_face_embedding"] is False


def test_markdown_labels_arcface_as_a_face_score():
    from nodes.core.drift import DriftReport

    report = DriftReport(scores=[0.9, 0.8], seqs=[1, 2], backend="arcface")
    md = report.markdown()
    assert "ArcFace" in md
    assert "Not a face-identity score" not in md
