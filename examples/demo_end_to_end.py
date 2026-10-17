#!/usr/bin/env python3
"""End-to-end demo: the whole spine, no GPU required.

    python examples/demo_end_to_end.py

    spec -> sheet prompt -> QC -> lock -> library -> mentions -> drift report

Two paths, and the demo says out loud which one ran:

**measured**
    If onnxruntime and the ArcFace models are present, the drift section
    embeds real public-domain portraits (three of Lincoln, two of Darwin, one
    of Curie) and reports the real cosine similarities. Same person scores
    high, different people score near zero. N is small because the repo ships
    six portraits, not six thousand; the numbers are still measured, not
    modelled.

**synthetic**
    Otherwise it falls back to a clearly labelled synthetic curve so the demo
    still runs on a bare machine. The fallback is labelled in the output and
    in the report itself, because a synthetic number that looks like a face
    score is worse than no number.

Everything except the synthetic fallback is the real code path.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nodes.core.drift import DRIFT_FLOOR, DriftReport, score_against_reference  # noqa: E402
from nodes.core.mentions import expand_mentions  # noqa: E402
from nodes.core.param_grid import (  # noqa: E402
    CharacterSpec,
    compile_bible,
    compile_prompt,
    spec_seed,
)
from nodes.core.sheet import build_sheet_prompt, qc_sheet, qc_sheet_metrics  # noqa: E402
from nodes.core.store import Element, ProjectStore, Shot  # noqa: E402

RULE = "=" * 66

FACES_DIR = Path(
    os.environ.get("SOULFOUNDRY_FACES_DIR", ROOT / "tests" / "fixtures" / "faces")
)

# Public-domain portraits shipped with the repo. Three of Lincoln give real
# same-person pairs across a decade; Darwin and Curie are different people.
SAME_PERSON = [("lincoln_b.jpg", "Lincoln, 1863"), ("lincoln_c.jpg", "Lincoln, 1860")]
OTHER_PEOPLE = [
    ("darwin_a.jpg", "Darwin, seated"),
    ("darwin_b.jpg", "Darwin, Cameron portrait"),
    ("curie_a.jpg", "Curie, 1920s"),
]


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _unit(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def synthetic_curve(n: int, start: float, step: float, seed0: int) -> list[float]:
    """Synthetic similarities, used only when no real embedder is available."""
    import random

    person = _unit([random.Random(7 * 1000 + i).uniform(-1, 1) for i in range(16)])
    scores = []
    for i in range(n):
        s = max(-1.0, min(1.0, start - i * step))
        other = (max(0.0, 1.0 - s * s)) ** 0.5
        shot = _unit(
            [random.Random((seed0 + i) * 7919 + k).uniform(-1, 1) for k in range(16)]
        )
        v = [s * p + other * o for p, o in zip(person, shot)]
        dot = sum(a * b for a, b in zip(person, v))
        na = sum(a * a for a in person) ** 0.5
        nb = sum(b * b for b in v) ** 0.5
        scores.append(dot / (na * nb))
    return scores


def real_measurement() -> dict | None:
    """Embed real portraits. Returns None when the embedder or fixtures are absent."""
    try:
        from nodes.core.embed import FaceEmbedder
    except Exception:
        return None

    embedder = FaceEmbedder()
    ok, why = embedder.available()
    if not ok:
        print(f"  (real embedder unavailable: {why})")
        return None

    needed = ["lincoln_a.jpg"] + [f for f, _ in SAME_PERSON + OTHER_PEOPLE]
    missing = [f for f in needed if not (FACES_DIR / f).is_file()]
    if missing:
        print(f"  (fixtures missing: {', '.join(missing)})")
        return None

    ref = embedder.embed(str(FACES_DIR / "lincoln_a.jpg"))
    if not ref.ok:
        print("  (no face detected in the reference portrait)")
        return None

    def collect(pairs):
        vecs, labels = [], []
        for name, label in pairs:
            r = embedder.embed(str(FACES_DIR / name))
            if r.ok:
                vecs.append(r.primary)
                labels.append(label)
        return vecs, labels

    locked_vecs, locked_labels = collect(SAME_PERSON)
    unlocked_vecs, unlocked_labels = collect(OTHER_PEOPLE)
    if not locked_vecs or not unlocked_vecs:
        return None

    return {
        "backend": "arcface",
        "dim": ref.dim,
        "reference_label": "Lincoln, 1862 (reference)",
        "locked": score_against_reference(ref.primary, locked_vecs),
        "locked_labels": locked_labels,
        "unlocked": score_against_reference(ref.primary, unlocked_vecs),
        "unlocked_labels": unlocked_labels,
    }


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="soulfoundry-demo-"))
    project = workdir / "noir"

    # --- 1. Declare the character -----------------------------------------
    header("1. CastParamGrid: declare the character once")
    spec = CharacterSpec(
        name="Santiago",
        genre="Western",
        budget=22,
        era="1900s",
        archetype="Rebel",
        gender="man",
        age="40s",
        build="Lean",
        height="tall",
        eye_color="Green",
        hair_style="Long",
        hair_texture="Wavy",
        hair_color="Auburn",
        facial_hair="Stubble",
        details="Scars",
        outfit="Workwear",
    )
    print(f"handle       : {spec.handle}")
    print(f"spec digest  : {spec.digest()}")
    print(f"budget band  : {spec.budget_band}")
    print(f"determinism  : seed {spec_seed(spec)} (same spec, same seed)")
    print(f"\nidentity clause:\n  {compile_prompt(spec)}")

    # --- 2. Build the sheet prompt ----------------------------------------
    header("2. SheetBuilder: the locking artifact prompt")
    sheet_prompt = build_sheet_prompt(compile_prompt(spec))
    print(f"sheet prompt is {len(sheet_prompt)} chars")
    print("lighting rules baked in:")
    for line in sheet_prompt.splitlines():
        if "5600K" in line or "catchlight" in line.lower() or "glare" in line:
            print(f"  {line.strip()}")

    # --- 3. QC a batch, pick the best -------------------------------------
    header("3. SheetQC: reject the lighting failures")
    batch = [
        {"panels": 3, "shadow_ratio": 0.71, "highlight_clip": 0.004,
         "catchlight": 0.31, "backdrop_delta": 5.0, "subject_coverage": 0.28},
        {"panels": 3, "shadow_ratio": 0.18, "highlight_clip": 0.061,
         "catchlight": 0.44, "backdrop_delta": 4.0, "subject_coverage": 0.31},
        {"panels": 3, "shadow_ratio": 0.19, "highlight_clip": 0.006,
         "catchlight": 0.42, "backdrop_delta": 4.0, "subject_coverage": 0.30},
    ]
    results = [qc_sheet_metrics(m) for m in batch]
    for i, result in enumerate(results, 1):
        verdict = "PASS" if result.passed else "REJECT"
        print(f"  candidate {i}: {verdict}  score {result.score:.2f}")
        for reason in result.reasons:
            print(f"      -> {reason}")
    best = max(range(len(results)), key=lambda i: (results[i].passed, results[i].score))
    print(f"\nkept candidate {best + 1} of {len(batch)} (batch-and-pick)")

    # --- 3b. QC a real IMAGE tensor ---------------------------------------
    header("3b. SheetQC on a real IMAGE tensor (the path that used to crash)")
    try:
        import numpy as np

        # A synthetic 3-panel sheet: grey backdrop, lit subject columns. The
        # point is the shape ([B,H,W,C] float 0..1) that ComfyUI actually
        # sends, not the pixels.
        sheet_arr = np.full((1, 320, 576, 3), 0.50, dtype=np.float32)
        for offset in (60, 252, 444):
            sheet_arr[:, 40:300, offset:offset + 84] = 0.78
        real_qc = qc_sheet(sheet_arr)
        print(f"  accepted [1,320,576,3] float tensor: score {real_qc.score:.2f}, "
              f"passed={real_qc.passed}")
        print("  checks: " + ", ".join(f"{k}={v}" for k, v in real_qc.checks.items()))
    except ImportError:
        print("  (numpy not installed; skipped)")

    # --- 4. Lock it into the store ----------------------------------------
    header("4. CharacterLock + ElementLibrary: persist the lock")
    bible = compile_bible(spec)
    with ProjectStore(project) as store:
        store.save_element(
            Element(
                name=spec.name,
                kind="character",
                spec_json=spec.to_json(),
                spec_digest=spec.digest(),
                sheet_path=str(workdir / "santiago-sheet.png"),
                bible_md=bible,
            )
        )
        store.pin(spec.name, True)
        store.save_element(
            Element(name="TheDocks", kind="location",
                    bible_md="Rain-slick timber docks at night, gas lamps, fog.")
        )
        store.save_element(
            Element(name="TheRevolver", kind="prop",
                    bible_md="Worn navy revolver, chipped walnut grip.")
        )
        summary = store.summary()
        print(f"store        : {store.path}")
        print(f"project      : {summary['project_id']}")
        print(f"elements     : {summary['elements']}")
        print("\nlibrary:")
        for el in store.list_elements():
            print(f"  {el.handle:<16} {el.kind:<10} "
                  f"{'pinned' if el.pinned else '      '} {el.spec_digest or '-'}")

        # --- 5. Mentions --------------------------------------------------
        header("5. MentionResolver: expand @name verbatim")
        shot_prompts = [
            "@santiago walks the length of @TheDocks, coat collar up.",
            "@santiago checks @TheRevolver, then holsters it.",
            "@santiago turns toward the lamp light, seen from behind.",
        ]
        library = store.library()
        for i, raw in enumerate(shot_prompts, 1):
            expanded = expand_mentions(raw, library, style="inline")
            print(f"  shot {i}: {len(raw)} -> {len(expanded)} chars")
            # The mention must be resolved. The bible documents its own handle
            # inside backticks, so strip code spans before asserting.
            prose = re.sub(r"`[^`]*`", "", expanded)
            assert not re.search(r"(?<![\w@])@[A-Za-z]", prose), (
                f"shot {i}: unresolved mention left in {prose[:80]!r}"
            )
            # Expanding an already-expanded prompt must be a no-op.
            assert expand_mentions(expanded, library, style="inline") == expanded

        # --- 6. Drift, locked vs unlocked ---------------------------------
        header("6. ConsistencyAudit: locked vs unlocked")
        measured = real_measurement()

        if measured:
            print(f"  embedder : {measured['backend']}, {measured['dim']}-d")
            print(f"  reference: {measured['reference_label']}")
            print(f"  locked   : {len(measured['locked'])} real frames of the same person")
            for label, score in zip(measured["locked_labels"], measured["locked"]):
                print(f"      {label:<28} {score:.3f}")
            print(f"  unlocked : {len(measured['unlocked'])} real frames of other people")
            for label, score in zip(measured["unlocked_labels"], measured["unlocked"]):
                print(f"      {label:<28} {score:.3f}")

            report = DriftReport(
                scores=measured["locked"],
                seqs=list(range(1, len(measured["locked"]) + 1)),
                baseline=measured["unlocked"],
                label="locked",
                backend=measured["backend"],
                notes=[
                    f"Measured with ArcFace on {len(measured['locked'])} "
                    f"same-person and {len(measured['unlocked'])} other-person "
                    f"public-domain portraits. Small N by design: these are the "
                    f"real photographs the repo ships, not a benchmark.",
                    "The baseline is the other-person set, which is what an "
                    "unlocked, re-prompted sequence drifts toward.",
                ],
            )
            path_used = "measured"
        else:
            print("  no real embedder available, falling back to a synthetic curve")
            print("  (this curve is labelled synthetic in the report)")
            n = 10
            report = DriftReport(
                scores=synthetic_curve(n, 0.97, 0.009, 100),
                seqs=list(range(1, n + 1)),
                baseline=synthetic_curve(n, 0.78, 0.012, 300),
                label="locked",
                backend="synthetic",
                notes=[
                    "SYNTHETIC: no face embedder was available, so these numbers "
                    "are generated, not measured. Do not publish them.",
                ],
            )
            path_used = "synthetic"

        for seq, score in zip(report.seqs, report.scores):
            store.save_shot(Shot(seq=seq, prompt=f"shot {seq}"))
            store.record_drift(seq, score)

        print()
        print(report.markdown())
        baseline_mean = sum(report.baseline) / len(report.baseline)
        print(f"\nunlocked baseline mean {baseline_mean:.3f} vs locked mean "
              f"{report.mean:.3f}  ->  lift {report.lift():+.3f}")
        print(f"floor {DRIFT_FLOOR}: holding = {report.holding}")
        print(f"curve persisted to the store: {store.drift_curve()}")

    # --- 7. Export --------------------------------------------------------
    header("7. CastSheetExport: portable, diffable output")
    export_dir = workdir / "export"
    export_dir.mkdir(exist_ok=True)
    bible_path = export_dir / f"{spec.name.lower()}-bible.md"
    bible_path.write_text(bible, encoding="utf-8")
    print(f"wrote {bible_path} ({len(bible)} chars)")

    header("DONE")
    if path_used == "measured":
        print("Drift numbers above are MEASURED with ArcFace on real photographs.")
    else:
        print("Drift numbers above are SYNTHETIC (no embedder on this machine).")
    print("Every other step is the real code path from nodes/core.")
    print(f"\nscratch dir: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
