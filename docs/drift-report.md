# Drift report

The differentiator: per-shot similarity against a locked reference, plus the
aggregate curve.

## Why it exists

Every character-consistency product claims the face holds. None of them
publish a number. So a buyer cannot tell a tool that holds for 4 shots from
one that holds for 40, and nobody can compare two approaches honestly.

`ConsistencyAudit` emits the number.

## How scoring works

1. Embed the locked reference (the sheet's face crop) with a face embedding
   model. This pack ships **ArcFace** (SCRFD detection, 5-point alignment to
   the canonical 112x112 template, MobileFaceNet backbone, 512-d), running on
   CPU through onnxruntime.
2. Embed the corresponding face crop from every generated shot the same way.
3. Cosine similarity of each shot against the reference.

Cosine, not Euclidean: embedding magnitude is an artifact of the extractor,
and only direction carries identity.

The backend is recorded on every report. `backend="arcface"` means real face
identity. Anything else (the opt-in perceptual descriptor used for CI) prints
`Not a face-identity score` at the top of the markdown and sets
`is_face_embedding: false` in `to_dict()`, so a descriptor can never be quoted
as a face match by accident.

Measured reference points from `tests/fixtures/faces`: two portraits of the
same person score 0.87 and 0.51 (different decades, one beardless);
unrelated people score 0.03, -0.01 and 0.09. That is the separation the floor
below is calibrated against.

## Reading the report

| Field | Meaning |
|---|---|
| `mean` | Average similarity across the sequence |
| `worst` / `worst_seq` | The weakest shot and where it happened |
| `holding` | True when no shot fell below the floor (default 0.75) |
| `slope` | Least-squares trend per shot. Negative means degrading |
| `degradation_seq` | First shot below the floor, if any |
| `baseline` / `lift` | The unlocked control, and the improvement over it |

## Thresholds

| Value | Meaning |
|---|---|
| `DRIFT_FLOOR = 0.75` | Below this it reads as a different person |
| `LOCK_CEILING = 0.90` | Above this the shots are effectively the same face |

Defaults, not laws. Face embedding scales differ between extractors, so
calibrate the floor against your own reference pairs before publishing
numbers.

## The control condition

A lift number needs a baseline, and the baseline must use the same prompts
and seeds with identity removed. That is what the `strip` mention style is
for:

```
ElementInjector(style="strip")   -> baseline frames
ElementInjector(style="inline")  -> locked frames
```

Same shots, one variable. Anything else is not a control.

## Honest reporting

The report is built to surface failure, not hide it. "0.94 mean, degrades to
0.88 by shot 9, here is the curve" is a stronger result than an unmeasured
claim of perfection, because it tells the reader exactly where the approach
stops working.

That is also why `holding` is a boolean against a published floor rather than
a vague "good" verdict, and why `degradation_seq` exists at all.

## Synthetic vs real numbers

`examples/demo_end_to_end.py` prints a full report from synthetic
embeddings, so the scoring path can be exercised with no GPU. Those numbers
demonstrate the math, not the model. Real numbers require running the actual
generation pipeline and embedding real faces.
