# SoulFoundry

Open-weights character consistency for ComfyUI. A project-scoped asset
library plus a **published drift score**, so "the same face across every
shot" stops being a claim and becomes a number.

Entry for the [ComfyUI Open Source Higgsfield challenge](https://comfy.org/open-source-higgsfield).

## The problem

Every character-consistency tool says the same thing: *the same character
across every scene*. None of them publish a measurement. You find out the
face drifted when you are 40 shots into a sequence.

SoulFoundry does what the closed tools do (lock identity into a reusable
reference first, instead of re-describing the character every shot) and then
measures the result. `ConsistencyAudit` embeds real frames with ArcFace and
emits per-shot similarity against the locked reference, a drift curve, and
the shot where it starts to fail.

A measured partial failure is a stronger result than an unmeasured claim.

## Measured, not modelled

The drift number comes from a real face embedder running on CPU, not from a
formula. From `python examples/demo_end_to_end.py`, on the six public-domain
portraits in `tests/fixtures/faces`:

```
  embedder : arcface, 512-d
  reference: Lincoln, 1862 (reference)
  locked   : 2 real frames of the same person
      Lincoln, 1863                0.870
      Lincoln, 1860                0.505
  unlocked : 3 real frames of other people
      Darwin, seated               0.028
      Darwin, Cameron portrait    -0.008
      Curie, 1920s                 0.091

unlocked baseline mean 0.037 vs locked mean 0.687  ->  lift +0.650
floor 0.75: holding = False
```

Two things worth reading carefully.

**The lift is real.** Same person scores 0.87; unrelated people score near
zero. That separation is the whole product.

**`holding = False` is the honest answer, not a bug.** The 1860 Lincoln
portrait is beardless and 0.505 falls under the floor, so the report says so.
That is exactly the behaviour the node exists for: it tells you which shot
drifted instead of promising perfection. If your sequence holds, it says
`yes`; if one shot fails, it names the shot.

N is small because the repo ships six photographs, not a benchmark set. The
numbers are measured, and the demo says which path produced them.

## Nodes

15 nodes. The first three are the sensor and the evidence; the rest are the
library.

| Node | Purpose |
|---|---|
| `CharacterEmbed` | **The sensor.** SCRFD + ArcFace on an IMAGE, 512-d vector, optional save |
| `EmbeddingFromFile` | Load a saved `.npy` vector plus its sidecar |
| `ConsistencyAudit` | **The differentiator.** Per-shot drift score, curve, first failing shot |
| `CastParamGrid` | Declare a character from the eight Soul Cast parameter categories |
| `SheetBuilder` | Compile a spec into a 3-panel turnaround prompt, lighting rules baked in |
| `SheetQC` / `SheetQCBatch` | Automated reject test, then pick the best of a batch |
| `CharacterLock` | Freeze the reference (embedding / LoRA / reference latent) |
| `ElementLibrary` | Project-scoped save, load, pin, delete, cross-project import |
| `ElementInjector` | Inject locked Elements into a shot prompt |
| `LocationElement` / `PropElement` | The location and prop analogues |
| `MentionResolver` | Expand `@name` to the full character bible, verbatim |
| `ShotListRunner` | Resolve a whole sequence in one pass |
| `CastSheetExport` | Export the bible as portable markdown or JSON |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DevvGwardo/soulfoundry
cd soulfoundry
pip install -r requirements-embed.txt   # numpy + onnxruntime, ~20 MB
python scripts/fetch_models.py          # ArcFace models, ~16 MB
```

ComfyUI already ships numpy. The only addition is `onnxruntime`, which is
what actually runs the face models, and it is CPU-only: no GPU, no torch, no
CUDA.

Without the models the node pack still loads. `CharacterEmbed` then raises a
clear error naming `scripts/fetch_models.py` rather than silently degrading,
because a number that looks like a face score but is not one is worse than an
error, and it ends up in a demo video.

## Try it with no GPU

```bash
python examples/demo_end_to_end.py
```

Runs the whole spine: spec, sheet prompt, QC and batch pick (including a real
`IMAGE` tensor through `qc_sheet`), lock, library, mention expansion, and a
locked-vs-unlocked drift report. With the models installed the drift numbers
are measured with ArcFace; without them the demo falls back to a synthetic
curve and labels it as synthetic in the output and in the report.

## Quick start

```python
from nodes.core.drift import build_report
from nodes.core.embed import embed_image, save_embedding
from nodes.core.param_grid import CharacterSpec, compile_bible
from nodes.core.store import Element, ProjectStore

# 1. the reference: one face, one vector
result = embed_image("reference.jpg")           # 512-d, L2-normalised
save_embedding(result.primary, "santiago_ref.npy", meta={"backend": result.backend})

# 2. the character, declared once
spec = CharacterSpec(name="santiago", genre="Western", archetype="Rebel", budget=20)

with ProjectStore("projects/noir") as store:
    store.save_element(Element(
        name=spec.name, kind="character",
        spec_json=spec.to_json(), spec_digest=spec.digest(),
        bible_md=compile_bible(spec), embed_path="santiago_ref.npy",
    ))

# 3. score generated frames against the lock
frames = [embed_image(p).primary for p in ["shot1.jpg", "shot2.jpg", "shot3.jpg"]]
report = build_report(result.primary, frames, label="locked", backend="arcface")
print(report.markdown())
```

In ComfyUI the same flow is node wiring: `CharacterEmbed` -> `CharacterLock`
-> `ConsistencyAudit`, with the generated frames going into the audit's
`frames` IMAGE input.

## The drift report

Real output from the demo (see above for the measured numbers):

```
# Drift report: locked

> Embedder: ArcFace (SCRFD detection, 5-point alignment, 512-d).

- Shots scored: **2**
- Mean similarity: **0.687**
- Worst shot: **0.505** (shot 2)
- Holding (no shot below 0.75): **no**
- Trend per shot: **-0.36476** (degrading)
- First shot below floor: **2**
- Unlocked baseline mean: **0.037**
- **Lift from locking: +0.650**
```

Every report carries the backend that produced it. A report from the
perceptual fallback says `Not a face-identity score` at the top, in the
markdown and in `to_dict()["is_face_embedding"]`, so it cannot be quoted as a
face-match result by accident.

## Design notes

**Identity is declared once, as data.** A `CharacterSpec` is JSON, diffable,
and hashable. The same spec compiles to the same prompt and the same seed
every run, which is what stops the character being reinvented between shots.

**The sensor is separate from the logic.** `nodes/core/embed.py` does image
normalisation, SCRFD detection, 5-point similarity alignment and the bilinear
warp in numpy, so it needs neither torch nor cv2 nor insightface. Everything
else in `nodes/core/` is stdlib.

**The strictness is deliberate.** `backend="auto"` resolves to ArcFace or
raises. It never quietly falls back to a descriptor that is not a face
embedding. The fallback exists for CI, is opt-in via `allow_perceptual=True`,
and labels itself everywhere.

**The bible is injected verbatim.** `@santiago` expands to the exact stored
text. An agent paraphrasing the description between shots is itself a drift
source, so expansion is a substitution, never a rewrite.

**Elements are three kinds.** Character, location, prop, matching the real
taxonomy. The library schema keys on it.

**The unlocked baseline is a real control.** Same prompts, same seeds,
identity stripped. Without it, a lift number means nothing.

## What is and is not wired

Honest scope, because this is a two-week build.

Wired and tested: face detection and embedding on real images, per-shot drift
scoring, image QC on real `IMAGE` tensors, the project store, mention
expansion, spec compilation and determinism.

Not wired: this pack does not *generate* video. It does not take a `MODEL`,
`CLIP` or `CONDITIONING` input and it does not steer a diffusion sampler. It
is the memory and the measurement. Generation comes from whichever model pack
you pair it with (LTX-2.5, Wan, or anything else), and the audit scores
whatever frames that produced.

Also not wired: training a LoRA (`CharacterLock` route B records the path,
another pack trains it) and extracting a reference latent (route C). Both are
recorded as artifacts with a spec digest so the lock stays verifiable.

## Tests

```bash
python -m pytest tests/ -q
```

158 tests, about 2 seconds, no GPU. The ArcFace tests run against real
portraits and skip cleanly when the models are absent. The suite covers the
ComfyUI contracts (socket types, `INPUT_TYPES` shape, `RETURN_TYPES` length)
and the normalisation cases that used to crash, including a duck-typed torch
tensor, because that is what an `IMAGE` socket actually carries.

## Layout

```
nodes/            ComfyUI node classes (thin wrappers)
nodes/core/       all real logic: params, sheet, store, mentions, drift, embed
tests/            unit tests + public-domain face fixtures
examples/         runnable end-to-end demo
scripts/          fetch_models.py
docs/             bible format, drift report format
store/            canonical SQL schema
```

## Not included, on purpose

Soul ID, training an identity from real photos of real people. That is the
deepfake-adjacent path, and submissions built on unlicensed likenesses are
disqualified rather than merely discouraged. SoulFoundry builds invented
characters only. The portraits in `tests/fixtures/faces` are public-domain
historical photographs used solely to verify that the embedder separates
people correctly.

## License

MIT for this repo. See [LICENSE](LICENSE).

**The face models are not MIT, and they are not in this repo.** `scripts/
fetch_models.py` downloads InsightFace's `buffalo_sc` pack (SCRFD detector +
ArcFace MobileFaceNet), which InsightFace licenses for non-commercial research
use. Redistributing those weights from an MIT repo would be a licence
violation, so the fetcher ships and the user downloads. For commercial use,
swap in a permissively-licensed embedder: `nodes/core/embed.py` isolates the
model behind `FaceEmbedder`, and only `detect()` and `embed()` touch ONNX, so
a replacement is one class, not a rewrite.

Worth knowing before submitting to a competition with an open-weights
requirement: the *code* is MIT and the *models* are research-licensed. If the
rules require every dependency to be openly licensed, use a different
embedder or note the distinction in the submission.
