"""Cast parameter grid: the eight Soul Cast categories, as data.

Open-weights equivalent of Higgsfield's Soul Cast parameter builder.
The point of this module is that identity is *declared once* as structured
data and then compiled deterministically into a prompt and a character
bible. Re-prompting from scratch per shot is what causes drift; a spec that
compiles the same way every time is the fix.

Pure Python. No ComfyUI, no torch.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

# --- Category 1: Genre (14) -------------------------------------------------

GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Thriller",
    "Horror",
    "Detective",
    "Romance",
    "Sci-Fi",
    "Fantasy",
    "War",
    "Western",
    "Historical",
    "Sitcom",
)

# --- Category 4: Archetype (12) --------------------------------------------

ARCHETYPES: tuple[str, ...] = (
    "Innocent",
    "Everyman",
    "Hero",
    "Caregiver",
    "Explorer",
    "Rebel",
    "Lover",
    "Creator",
    "Jester",
    "Sage",
    "Magician",
    "Ruler",
)

# --- Category 3: Era --------------------------------------------------------

ERAS: tuple[str, ...] = (
    "1900s",
    "1910s",
    "1920s",
    "1930s",
    "1940s",
    "1950s",
    "1960s",
    "1970s",
    "1980s",
    "1990s",
    "2000s",
    "2010s",
    "2020s",
    "Contemporary",
    "Near-Future",
    "Far-Future",
)

# --- Category 8: Outfit -----------------------------------------------------

OUTFITS: tuple[str, ...] = (
    "Casual",
    "Formal",
    "High Fashion",
    "Military",
    "Sporty",
    "Workwear",
    "Vintage",
)

# --- Category 7: Details ----------------------------------------------------

DETAILS: tuple[str, ...] = (
    "None",
    "Scars",
    "Freckles",
    "Tattoos",
    "Eye Patch",
    "Manual",
)

# --- Category 2: Budget -----------------------------------------------------
# A slider. Low reads gritty indie, high reads blockbuster polish. Rendered
# into the prompt as photographic language, since that is what actually
# moves a diffusion model.

BUDGET_PRESETS: dict[str, str] = {
    "gritty_indie": (
        "16mm film grain, natural available light, slightly soft focus, "
        "documentary realism, muted desaturated palette"
    ),
    "streaming": (
        "digital cinema capture, clean modern grade, balanced contrast, "
        "shallow depth of field"
    ),
    "blockbuster": (
        "large-format anamorphic capture, high dynamic range, crisp "
        "edge-to-edge detail, rich saturated palette, cinematic key light"
    ),
}

# --- Category 6: Physical appearance vocab ---------------------------------

HAIR_STYLES: tuple[str, ...] = (
    "Cropped",
    "Buzz",
    "Short",
    "Bob",
    "Pixie",
    "Shoulder-Length",
    "Long",
    "Braided",
    "Coiled",
    "Tied Back",
    "Shaved",
)

HAIR_TEXTURES: tuple[str, ...] = (
    "Straight",
    "Wavy",
    "Curly",
    "Coily",
    "Fine",
    "Thick",
)

HAIR_COLORS: tuple[str, ...] = (
    "Black",
    "Dark Brown",
    "Brown",
    "Auburn",
    "Ginger",
    "Blonde",
    "Platinum",
    "Silver",
    "Grey",
    "Dyed",
)

EYE_COLORS: tuple[str, ...] = (
    "Dark Brown",
    "Brown",
    "Hazel",
    "Green",
    "Blue",
    "Grey",
    "Amber",
)

BUILDS: tuple[str, ...] = (
    "Slight",
    "Lean",
    "Average",
    "Athletic",
    "Stocky",
    "Broad",
    "Heavyset",
)

FACIAL_HAIR: tuple[str, ...] = (
    "None",
    "Stubble",
    "Moustache",
    "Goatee",
    "Full Beard",
    "Long Beard",
)


def budget_band(budget: int) -> str:
    """Map the 0-100 budget slider onto a named photographic band."""
    if budget < 34:
        return "gritty_indie"
    if budget < 67:
        return "streaming"
    return "blockbuster"


@dataclass
class CharacterSpec:
    """A character, declared once, as structured data.

    This is the object ``CastParamGrid`` emits as ``CHARACTER_SPEC`` and the
    object every later node in the pack consumes. It is deliberately plain
    data: JSON-serialisable, diffable, and hashable so that a lock can be
    verified against the spec that produced it.
    """

    # 1. Genre
    genre: str = "Drama"
    # 2. Budget (0-100 slider)
    budget: int = 50
    # 3. Era
    era: str = "Contemporary"
    # 4. Archetype
    archetype: str = "Everyman"
    # 5. Identity
    gender: str = "unspecified"
    race_ethnicity: str = "unspecified"
    age: str = "30s"
    # 6. Physical appearance
    build: str = "Average"
    height: str = "average height"
    eye_color: str = "Brown"
    hair_style: str = "Short"
    hair_texture: str = "Straight"
    hair_color: str = "Dark Brown"
    facial_hair: str = "None"
    # 7. Details
    details: str = "None"
    details_manual: str = ""
    # 8. Outfit
    outfit: str = "Casual"
    outfit_notes: str = ""
    # Free-text additions
    name: str = "unnamed"
    notes: str = ""
    # Auto-generated, stored so the bible is stable across runs
    backstory: str = ""
    motivation: str = ""
    fear: str = ""
    flaw: str = ""
    strength: str = ""

    def __post_init__(self) -> None:
        self.validate()

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []
        if self.genre not in GENRES:
            errors.append(f"genre {self.genre!r} not in {GENRES}")
        if self.archetype not in ARCHETYPES:
            errors.append(f"archetype {self.archetype!r} not in {ARCHETYPES}")
        if self.era not in ERAS:
            errors.append(f"era {self.era!r} not in {ERAS}")
        if self.outfit not in OUTFITS:
            errors.append(f"outfit {self.outfit!r} not in {OUTFITS}")
        if self.details not in DETAILS:
            errors.append(f"details {self.details!r} not in {DETAILS}")
        if not isinstance(self.budget, int) or not 0 <= self.budget <= 100:
            errors.append(f"budget {self.budget!r} must be an int in 0..100")
        if self.details == "Manual" and not self.details_manual.strip():
            errors.append("details='Manual' requires details_manual to be set")
        if errors:
            raise ValueError("invalid CharacterSpec: " + "; ".join(errors))

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterSpec:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown CharacterSpec fields: {sorted(unknown)}")
        return cls(**data)

    def digest(self) -> str:
        """Stable content hash of the spec.

        Used to prove a locked Element still matches the spec that made it,
        and to make generations reproducible.
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    @property
    def handle(self) -> str:
        """The ``@mention`` handle this character is called by.

        Normalised to ASCII: handles end up inside prompts and file names, so
        a non-ASCII slug would break both.
        """
        decomposed = unicodedata.normalize("NFKD", self.name)
        ascii_name = decomposed.encode("ascii", "ignore").decode("ascii")
        slug = "".join(ch for ch in ascii_name.lower() if ch.isalnum())
        return f"@{slug or 'unnamed'}"

    @property
    def budget_band(self) -> str:
        return budget_band(self.budget)

    @property
    def budget_language(self) -> str:
        return BUDGET_PRESETS[self.budget_band]


def spec_seed(spec: CharacterSpec) -> int:
    """Deterministic RNG seed derived from the spec.

    Same spec, same seed, same sheet. That reproducibility is the difference
    between a character that holds and one that gets reinvented.
    """
    return int(spec.digest(), 16) % (2**31)


# --- Compilation -----------------------------------------------------------


def _appearance_clause(spec: CharacterSpec) -> str:
    parts: list[str] = []
    if spec.gender and spec.gender != "unspecified":
        parts.append(spec.gender)
    if spec.age:
        parts.append(spec.age)
    if spec.race_ethnicity and spec.race_ethnicity != "unspecified":
        parts.append(spec.race_ethnicity)
    parts.append(f"{spec.build.lower()} build")
    if spec.height:
        parts.append(spec.height)
    parts.append(f"{spec.eye_color.lower()} eyes")
    parts.append(
        f"{spec.hair_color.lower()} {spec.hair_texture.lower()} "
        f"{spec.hair_style.lower()} hair"
    )
    if spec.facial_hair and spec.facial_hair != "None":
        parts.append(f"{spec.facial_hair.lower()}")
    return ", ".join(parts)


def _detail_clause(spec: CharacterSpec) -> str:
    if spec.details == "None":
        return ""
    if spec.details == "Manual":
        return spec.details_manual.strip()
    return spec.details.lower()


def compile_prompt(spec: CharacterSpec) -> str:
    """Compile a spec into the identity clause used in every generation.

    Deliberately verbatim and order-stable. This string is the thing that
    gets injected into shot prompts by ``ElementInjector`` and expanded from
    ``@name`` by ``MentionResolver``. It is never paraphrased.
    """
    clauses = [
        f"{spec.era} {spec.genre.lower()} {spec.archetype.lower()}",
        _appearance_clause(spec),
        f"wearing {spec.outfit.lower()}"
        + (f", {spec.outfit_notes.strip()}" if spec.outfit_notes.strip() else ""),
    ]
    detail = _detail_clause(spec)
    if detail:
        clauses.append(detail)
    clauses.append(spec.budget_language)
    if spec.notes.strip():
        clauses.append(spec.notes.strip())
    return ". ".join(c.rstrip(". ") for c in clauses if c) + "."


def _derived_backstory(spec: CharacterSpec) -> dict[str, str]:
    """Deterministic story beats from genre + archetype.

    Not an LLM call: a lookup, so the bible is identical every run and can be
    asserted in tests. An optional LLM refinement pass is a later feature.
    """
    beats: dict[str, dict[str, str]] = {
        "Hero": {
            "motivation": "to protect the people who cannot protect themselves",
            "fear": "failing the ones who trusted them",
            "flaw": "mistakes self-sacrifice for virtue",
            "strength": "unshakeable resolve under pressure",
        },
        "Everyman": {
            "motivation": "to get back to an ordinary life",
            "fear": "being nobody in particular",
            "flaw": "avoids conflict until it is unavoidable",
            "strength": "endless, unglamorous resilience",
        },
        "Rebel": {
            "motivation": "to tear down a system that deserves it",
            "fear": "becoming the thing they oppose",
            "flaw": "mistakes destruction for progress",
            "strength": "refuses to be intimidated",
        },
        "Sage": {
            "motivation": "to understand how it all actually fits together",
            "fear": "being wrong about something that mattered",
            "flaw": "detached when presence is what is needed",
            "strength": "sees the pattern before anyone else",
        },
        "Caregiver": {
            "motivation": "to keep everyone in their orbit safe",
            "fear": "being needed by no one",
            "flaw": "gives until there is nothing left",
            "strength": "absolute reliability",
        },
        "Explorer": {
            "motivation": "to see what is over the next ridge",
            "fear": "dying somewhere they have already been",
            "flaw": "leaves before anything can take root",
            "strength": "adapts to any terrain or culture",
        },
        "Ruler": {
            "motivation": "to impose order on a chaotic world",
            "fear": "losing control of the room",
            "flaw": "confuses loyalty with obedience",
            "strength": "commands a room without raising their voice",
        },
        "Creator": {
            "motivation": "to make the thing only they can see",
            "fear": "being ordinary",
            "flaw": "sacrifices people to the work",
            "strength": "turns constraints into style",
        },
        "Innocent": {
            "motivation": "to keep the world as good as they believe it is",
            "fear": "finding out it is not",
            "flaw": "trusts the wrong person every time",
            "strength": "restores everyone else's faith",
        },
        "Lover": {
            "motivation": "to be fully known by one person",
            "fear": "being left",
            "flaw": "loses themselves in the other",
            "strength": "makes people feel seen",
        },
        "Jester": {
            "motivation": "to keep the room from going dark",
            "fear": "silence",
            "flaw": "jokes through the moment that needed honesty",
            "strength": "deflates tyrants with one line",
        },
        "Magician": {
            "motivation": "to make the impossible look easy",
            "fear": "being seen through",
            "flaw": "hoards what they know",
            "strength": "reframes any problem into a solvable one",
        },
    }
    beat = beats.get(spec.archetype, beats["Everyman"])
    origin = {
        "War": "made by a war they did not start",
        "Western": "made by a frontier that does not forgive",
        "Sci-Fi": "made by a system that outgrew its makers",
        "Fantasy": "made by an old debt to something older",
        "Horror": "made by something they will not name",
        "Romance": "made by a choice they cannot take back",
        "Detective": "made by a case that never closed",
        "Historical": "made by a century that had no room for them",
    }.get(
        spec.genre,
        "made by an ordinary life that quietly stopped being ordinary",
    )
    return {
        "backstory": (
            f"A {spec.age} {spec.archetype.lower()} in a {spec.era} "
            f"{spec.genre.lower()} world, {origin}. Wears "
            f"{spec.outfit.lower()}; reads as {spec.build.lower()} and "
            f"carries the {spec.hair_color.lower()} hair of someone who "
            f"stopped explaining themselves."
        ),
        **beat,
    }


def compile_bible(spec: CharacterSpec) -> str:
    """Compile a spec into the portable character bible (markdown).

    This is the artifact that gets injected verbatim when ``@name`` appears
    in a prompt, and the artifact exported by ``CastSheetExport``. Keeping it
    markdown means it is diffable in git.
    """
    derived = _derived_backstory(spec)
    story = {
        "backstory": spec.backstory or derived["backstory"],
        "motivation": spec.motivation or derived["motivation"],
        "fear": spec.fear or derived["fear"],
        "flaw": spec.flaw or derived["flaw"],
        "strength": spec.strength or derived["strength"],
    }
    lines = [
        f"# {spec.name}",
        "",
        f"- **Handle:** `{spec.handle}`",
        f"- **Spec digest:** `{spec.digest()}`",
        f"- **Archetype:** {spec.archetype}",
        f"- **Genre / Era:** {spec.genre} / {spec.era}",
        f"- **Budget band:** {spec.budget_band} ({spec.budget}/100)",
        "",
        "## Identity prompt",
        "",
        "```text",
        compile_prompt(spec),
        "```",
        "",
        "## Physical",
        "",
        f"- Build: {spec.build}",
        f"- Height: {spec.height}",
        f"- Eyes: {spec.eye_color}",
        f"- Hair: {spec.hair_color}, {spec.hair_texture}, {spec.hair_style}",
        f"- Facial hair: {spec.facial_hair}",
        f"- Details: {_detail_clause(spec) or 'none'}",
        "",
        "## Wardrobe",
        "",
        f"- {spec.outfit}"
        + (f" - {spec.outfit_notes}" if spec.outfit_notes.strip() else ""),
        "",
        "## Story",
        "",
        f"- **Backstory:** {story['backstory']}",
        f"- **Motivation:** {story['motivation']}",
        f"- **Fear:** {story['fear']}",
        f"- **Flaw:** {story['flaw']}",
        f"- **Strength:** {story['strength']}",
        "",
        "---",
        "",
        "_Generated by SoulFoundry. This block is injected verbatim; do not "
        "paraphrase it._",
    ]
    return "\n".join(lines)
