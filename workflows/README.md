# Workflows

Three graphs make up the submission. Each is described here as a node chain;
the API-format JSON is exported from ComfyUI once the models are wired, since
model filenames and node ids are environment-specific.

## 1. `build_character_sheet`

Generate a batch of sheets, auto-reject the failures, keep the best.

```
CastParamGrid ──> SheetBuilder ──> <text-to-image> ──> SheetQC ──┐
      │                                                            │
      └──> (spec)                                                 │
                                                                  v
                                                    SheetQCBatch (pick best)
                                                                  │
                                                                  v
                                                          CharacterLock ──> ElementLibrary (save)
```

Notes:
- Generate 4 to 8 candidates from the same spec. `SheetBuilder` emits a
  deterministic seed, so pass distinct seeds to get distinct candidates.
- `SheetQC` rejects on panel count, shadow ratio, hair glare, missing
  catchlight, backdrop inconsistency and cropped feet. Most failures are
  lighting failures, not likeness failures.
- Keep two finalists, not one. A sheet that reads well as a still can fall
  apart in motion, and you cannot tell which holds until you test it.

## 2. `lock_character`

Turn an approved sheet into a persistent Element. `CharacterEmbed` is the
sensor: it runs SCRFD + ArcFace on the sheet's face and emits a 512-d vector,
which `CharacterLock` persists next to the project.

```
LoadImage ──> CharacterEmbed ──> CharacterLock (route=embedding) ──> ElementLibrary (action=save, pin)
                    │                        ▲
                    └── embedding ───────────┘
```

Notes:
- `CharacterEmbed` saves to `<project>/embeddings/<name>_reference.npy` plus a
  JSON sidecar recording the backend, so a vector can never be mistaken for
  one from a different embedder later.
- `EmbeddingFromFile` loads a saved vector back, so an audit is reproducible
  without re-embedding the reference every run.
- Routes B (LoRA) and C (reference latent) record an artifact path instead.
  This pack does not train a LoRA; another pack does, and `CharacterLock`
  records the path plus the spec digest so the lock stays verifiable.

Route A (`embedding`) needs no training and is the default. Route B (`lora`)
is the high-fidelity upgrade and needs 10 to 30 varied images plus GPU time.
Route C (`reference_latent`) feeds reference-to-video, which is the 2026
default for multi-shot video.

## 3. `multi_shot_sequence`

Run a whole sequence with the lock applied.

```
shots.json ──> ShotListRunner ──> per-shot prompt
                                        │
                                        v
                     ElementInjector (style=inline)  ──> <video model> ──> frames
                                        │                                      │
                                        └── style=strip ──> <video model> ──> baseline frames
                                                                               │
                                                                               v
                                                          ConsistencyAudit ──> drift report
```

The `strip` branch is the control condition. Same prompts, same seeds,
identity removed. Without it a lift number means nothing.

## App Mode

The competition asks for a usable interface, not just nodes. The plan is Comfy
App Mode / App Builder: expose the character picker, the shot prompt and a
camera note, hide the graph, ship it as one shareable URL running on Comfy
Cloud.

See `app/README.md`.
