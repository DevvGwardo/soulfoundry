# Phase 0 checklist

Everything here needs an account, a credential, or a decision that only you
can make. The repo is scaffolded and the code is tested; these are the
external gates.

Deadline: **October 30, 9:00am PT**. Build window opened October 15. Winners
November 6.

---

## 1. Comfy Cloud / API access

Comfy API build and deployment limits start at **Standard**: $20/month
monthly, or $16/month billed yearly ($192/year). The free tier gives 5 runs
on real GPUs but no API builds.

- [ ] Subscribe to Standard at https://comfy.org/pricing
- [ ] Create an API key
- [ ] Export it: `export COMFY_API_KEY="comfyui-..."`
- [ ] Run the SDK round trip once and confirm it works

```bash
pip install comfy-sdk
python - <<'EOF'
from comfy_sdk import Comfy
client = Comfy(api_key="comfyui-...")
wf = client.workflows.from_file("workflow_api.json")
job = client.run(wf)
print(job)
EOF
```

Do not skip this. If the round trip is not proven on day one, every later
estimate is fiction.

---

## 2. Comfy Registry publisher

The publisher id is globally unique and **cannot be changed** after creation.
It lives in the node URL, so pick it deliberately.

- [ ] Create a publisher at https://registry.comfy.org/
- [ ] Create a publishing API key for it
- [ ] Put the id in `pyproject.toml` under `[tool.comfy] PublisherId`
- [ ] Add it as the `REGISTRY_ACCESS_TOKEN` repo secret for CI publishing

The display name in `pyproject.toml` is currently `SoulFoundry`. Fallbacks if
that is taken on the registry: `ElementForge`, `CastLock`.

---

## 3. LTX-2.5 weights (gated)

The weights are gated on Hugging Face. Downloads fail without approval, and
approval is not instant.

- [ ] Request access to `Lightricks/LTX-2.5` on Hugging Face
- [ ] Wait for approval

Do this early, not the night before submission.

---

## 4. Decide the embedding route

`CharacterLock` supports three routes. The default is `embedding`, which
needs no training. Pick the concrete node pack you will actually run:

- [ ] Choose a face-embedding path: IP-Adapter, PuLID, InstantID, or
      PhotoMaker
- [ ] Choose a video model: LTX-2.5 is the strategic pick (open weights, and
      the LTX bonus stacks with the Open Weights Only prize)

Note the competition's Q&A confirms partner models are allowed outside the
Open Weights Only category. The one genuinely unresolved question is whether
running open weights through Comfy Cloud satisfies that category, or whether
it requires a self-hosted deployment. Worth asking in `#developer-platform`
on the Comfy Discord before committing architecture.

---

## 5. Repo settings

- [ ] Decide public or private. The rules require a GitHub repo with an open
      license and run instructions; public is the safe reading
- [ ] Add the CI workflows. They are written and on disk at
      `.github/workflows/` but were **not pushed**: the current GitHub token
      lacks the `workflow` scope, so GitHub rejects them. Either add the scope
      and re-push, or add the files through the web UI
- [ ] Upload the icon. `assets/icon.png` is generated and is 800x400, which
      is the registry's stated maximum

---

## 6. Submission deliverables

Four required, all of them:

- [ ] GitHub repo, open license, run instructions
- [ ] 2-minute demo video
- [ ] Social post tagged `#OpenSourceHiggsfield`
- [ ] Typeform submitted

One entry per person. Do not split effort across two ideas.

---

## What is already done

Everything below is committed and verified on a clean clone:

- 13 ComfyUI node classes, all registered and contract-tested
- 134 tests, passing in about a second with no GPU
- The full spine working end to end: parameter grid, sheet prompt with the
  lighting rules baked in, QC reject test, batch pick, lock, project store,
  mention expansion, and the drift report
- `examples/demo_end_to_end.py`, which runs the whole pipeline with synthetic
  embeddings so it works anywhere
- README, MIT license, SQL schema, docs for the bible format and the drift
  report, `pyproject.toml`, `.comfyignore`

The demo currently reports a 0.936 mean degrading to 0.890 by shot 9, against
an unlocked baseline of 0.756. Those numbers are from synthetic embeddings:
they exercise the math, not the model. Real numbers need the real pipeline.
