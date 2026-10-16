#!/usr/bin/env python3
"""End-to-end demo: the minimum viable spine, no GPU required.

Runs the whole pipeline with synthetic data so it can execute anywhere:

    spec -> sheet prompt -> QC -> lock -> library -> mentions -> drift report

    python examples/demo_end_to_end.py

Everything here is the real code path. Only the embeddings are synthetic,
because a face embedding model needs a GPU and a model download.
"""

from __future__ import annotations

import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes.core.drift import DRIFT_FLOOR, build_report
from nodes.core.mentions import expand_mentions
from nodes.core.param_grid import CharacterSpec, compile_bible, compile_prompt, spec_seed
from nodes.core.sheet import build_sheet_prompt, qc_sheet_metrics
from nodes.core.store import Element, ProjectStore, Shot

RULE = "=" * 66


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _unit(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def fake_embedding(seed: int, similarity: float, *, base_seed: int = 7) -> list[float]:
    """A synthetic 8-dim 'face embedding' with a *known* cosine to the reference.

    Stands in for a real face embedder so the demo runs without a GPU.

    Cosine similarity is scale-invariant, so scaling a vector does not move
    it. To control similarity directly, blend the reference direction with an
    independent per-shot direction:

        v = s * unit(base) + sqrt(1 - s^2) * unit(noise)

    In 8 dimensions the noise direction is near-orthogonal to the base, so
    ``cos(v, base) ~= s``. That makes the demo curve a real measurement
    rather than a number that happens to print.
    """
    base_rng = random.Random(base_seed)
    noise_rng = random.Random(seed)
    base = _unit([base_rng.uniform(-1, 1) for _ in range(8)])
    noise = _unit([noise_rng.uniform(-1, 1) for _ in range(8)])
    s = max(-1.0, min(1.0, similarity))
    other = (max(0.0, 1.0 - s * s)) ** 0.5
    return [s * b + other * n for b, n in zip(base, noise)]


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
                embed_path=str(workdir / "santiago-ipadapter.pt"),
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
        print(f"\nlibrary:\n{store.list_elements() and ''}", end="")
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
        locked = build_report(
            reference=fake_embedding(0, 1.0),
            frames=[fake_embedding(100 + i, 0.97 - i * 0.008) for i in range(10)],
            label="locked",
        )
        unlocked = build_report(
            reference=fake_embedding(0, 1.0),
            frames=[fake_embedding(200 + i, 0.80 - i * 0.02) for i in range(10)],
            label="unlocked",
        )
        locked.baseline = unlocked.scores

        for seq, score in zip(locked.seqs, locked.scores):
            store.save_shot(Shot(seq=seq, prompt=f"shot {seq}"))
            store.record_drift(seq, score)

        print(locked.ascii_curve())
        print()
        print(locked.markdown())
        print(f"\nunlocked mean {unlocked.mean:.3f} vs locked mean "
              f"{locked.mean:.3f}  ->  lift {locked.lift():+.3f}")
        print(f"floor {DRIFT_FLOOR}: locked holding = {locked.holding}, "
              f"unlocked holding = {unlocked.holding}")
        print(f"curve persisted to the store: {store.drift_curve()}")

    # --- 7. Export --------------------------------------------------------
    header("7. CastSheetExport: portable, diffable output")
    export_dir = workdir / "export"
    export_dir.mkdir(exist_ok=True)
    bible_path = export_dir / f"{spec.name.lower()}-bible.md"
    bible_path.write_text(bible, encoding="utf-8")
    print(f"wrote {bible_path} ({len(bible)} chars)")

    header("DONE")
    print("Every step above is the real code path from nodes/core.")
    print("Only the embeddings are synthetic: a face embedder needs a GPU.")
    print(f"\nscratch dir: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
