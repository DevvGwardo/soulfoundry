# App interface

The competition requires a working creative app with a usable interface, not
just a node pack. This is the plan for it.

## Approach: Comfy App Mode

Comfy ships App Mode and App Builder: mark node inputs as application
parameters, hide the graph, and the result runs as a single shareable URL on
Comfy Cloud. That is less work than a bespoke front end and it keeps the whole
thing on the Comfy Developer Platform.

## What the user sees

Four surfaces, and nothing else:

1. **Cast** - pick an archetype, genre, era and outfit from the parameter
   grid. Everything else stays at defaults unless expanded.
2. **Library** - the project's Elements, with their `@handle` and a pin
   toggle. The working set for the current sequence.
3. **Shot list** - rows of prompt text with `@name` mentions. One row per
   shot, numbered.
4. **Drift report** - the curve, the mean, and the shot where it starts to
   fail.

Everything the director does not care about stays hidden: the embedding
nodes, the sampler settings, the seed plumbing.

## Why this shape

The differentiator is the drift report, and it is only legible if the other
three surfaces are simple. A graph-shaped UI would bury it.

## Status

Not built yet. The nodes it would drive are implemented and tested; the App
Mode export needs a Comfy Cloud deployment to publish against.
