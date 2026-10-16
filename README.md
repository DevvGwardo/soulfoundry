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
measures the result. `ConsistencyAudit` emits per-shot similarity against
the locked reference, a drift curve, and the shot where it starts to fail.

A measured partial failure is a stronger result than an unmeasured claim.

## Nodes

| Node | Purpose |
|---|---|
| `CastParamGrid` | Declare a character from the eight Soul Cast parameter categories |
| `SheetBuilder` | Compile a spec into a 3-panel turnaround prompt, lighting rules baked in |
| `SheetQC` / `SheetQCBatch` | Automated reject test, then pick the best of a batch |
| `CharacterLock` | Freeze the reference (embedding / LoRA / reference latent) |
| `ElementLibrary` | Project-scoped save, load, pin, delete, cross-project import |
| `ElementInjector` | Inject locked Elements into a shot prompt |
| `LocationElement` / `PropElement` | The location and prop analogues |
| `MentionResolver` | Expand `@name` to the full character bible, verbatim |
| `ConsistencyAudit` | **The differentiator.** Per-shot drift score and curve |
| `ShotListRunner` | Resolve a whole sequence in one pass |
| `CastSheetExport` | Export the bible as portable markdown or JSON |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DevvGwardo/soulfoundry
```

No Python dependencies for the core. `qc_sheet(image=...)` needs Pillow and
numpy; everything else is stdlib.

## Try it with no GPU

```bash
python examples/demo_end_to_end.py
```

Runs the whole spine: spec, sheet prompt, QC and batch pick, lock, library,
mention expansion, and a locked-vs-unlocked drift report. Only the face
embeddings are synthetic, because a real embedder needs a GPU and a model
download.

## Quick start

```python
from nodes.core.param_grid import CharacterSpec, compile_bible
from nodes.core.sheet import build_sheet_prompt, qc_sheet_metrics
from nodes.core.store import ProjectStore, Element
from nodes.core.drift import build_report

spec = CharacterSpec(name="santiago", genre="Western", archetype="Rebel", budget=20)

identity = compile_bible(spec)
prompt = build_sheet_prompt(identity)

with ProjectStore("projects/noir") as store:
    store.save_element(Element(
        name=spec.name, kind="character",
        spec_json=spec.to_json(), spec_digest=spec.digest(), bible_md=identity,
    ))

report = build_report(reference=[1.0, 0.0], frames=[[1.0, 0.0], [0.9, 0.1]])
print(report.markdown())
```

## The drift report

```
# Drift report: locked

- Shots scored: **10**
- Mean similarity: **0.939**
- Worst shot: **0.914** (shot 7)
- Holding (no shot below 0.75): **yes**
- Trend per shot: **-0.00423** (degrading)
- Unlocked baseline mean: **0.729**
- **Lift from locking: +0.211**
```

## Design notes

**Identity is declared once, as data.** A `CharacterSpec` is JSON, diffable,
and hashable. The same spec compiles to the same prompt and the same seed
every run, which is what stops the character being reinvented between shots.

**Core logic is ComfyUI-free.** Everything testable lives in `nodes/core/`.
The ComfyUI classes in `nodes/` are thin shells: validate, call core, return.
That is why the test suite runs in about a second with no GPU.

**The bible is injected verbatim.** `@santiago` expands to the exact stored
text. An agent paraphrasing the description between shots is itself a drift
source, so expansion is a substitution, never a rewrite.

**Elements are three kinds.** Character, location, prop, matching the real
taxonomy. The library schema keys on it.

**The unlocked baseline is a real control.** Same prompts, same seeds,
identity stripped. Without it, a lift number means nothing.

## Tests

```bash
python -m pytest tests/ -q
```

## Layout

```
nodes/            ComfyUI node classes (thin wrappers)
nodes/core/       all real logic: params, sheet, store, mentions, drift
tests/            unit tests, no GPU required
examples/         runnable end-to-end demo
docs/             bible format, drift report format
store/            canonical SQL schema
```

## Not included, on purpose

Soul ID, training an identity from real photos of real people. That is the
deepfake-adjacent path, and submissions built on unlicensed likenesses are
disqualified rather than merely discouraged. SoulFoundry builds invented
characters only.

## License

MIT. See [LICENSE](LICENSE).
