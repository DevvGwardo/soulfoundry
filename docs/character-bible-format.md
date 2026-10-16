# Character bible format

The bible is the artifact injected verbatim wherever `@name` appears. It is
markdown so it diffs in git.

## Structure

```
# <name>

- **Handle:** `@handle`
- **Spec digest:** `<16 hex chars>`
- **Archetype:** ...
- **Genre / Era:** ... / ...
- **Budget band:** ... (n/100)

## Identity prompt
<the compiled one-paragraph identity clause>

## Physical
- Build / Height / Eyes / Hair / Facial hair / Details

## Wardrobe
- Outfit

## Story
- Backstory / Motivation / Fear / Flaw / Strength
```

## Rules

**Never paraphrase it.** The point of storing it is that the exact words get
reused. If a generation needs different wording, change the spec and re-lock,
so the digest changes too.

**The digest is the contract.** `spec.digest()` is a sha256 of the canonical
JSON of the spec, truncated to 16 hex characters. Two locks with the same
digest came from the same declared character. If a shot looks wrong and the
digest matches, the problem is the model or the seed, not the description.

**Story is derived, not invented at generation time.** `compile_bible` fills
backstory, motivation, fear, flaw and strength deterministically from
archetype and genre unless they were set explicitly on the spec. Two runs of
the same spec produce byte-identical bibles, which is what makes them
testable.

**Handles are ASCII.** Lowercased, alphanumeric. `Old Man River` becomes
`@oldmanriver`, `Éowyn` becomes `@eowyn`. Handles end up inside prompts and
file names, so non-ASCII would break both.

**The handle is documented inside backticks.** The bible writes its own
handle as `` `@santiago` ``, which is prose *about* the handle, not a
reference to resolve. The mention regex excludes backtick-adjacent text so
expanding a prompt cannot re-inject a mention from the bible's own header.
That is what keeps expansion idempotent.

## Mention expansion styles

| Style | What it does | When |
|---|---|---|
| `inline` | Substitutes the bible in place, flattened to one line, truncated with `…` past `max_inline_chars` | The prompt has a length budget |
| `append` | Keeps `@name` and adds a delimited reference block at the end | Video models that weight early tokens heavily |
| `strip` | Removes mentions entirely, no bible | The **unlocked control condition** for the drift report |

`strip` is how the baseline gets measured: same prompts, same seeds, identity
removed. That gives the drift number without the lock.

## Export

`CastSheetExport` writes `markdown`, `json`, or `prompt`. The JSON export
carries `_compile_prompt` and `_digest` so a downstream tool does not have to
re-derive them.

Free and portable. The closed equivalent charges a per-character fee to
"lock" a character and does not let the trained identity leave the platform.
