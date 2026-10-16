"""Tests for the cast parameter grid and compilation."""

from __future__ import annotations

import pytest

from nodes.core.param_grid import (
    ARCHETYPES,
    ERAS,
    GENRES,
    OUTFITS,
    CharacterSpec,
    budget_band,
    compile_bible,
    compile_prompt,
    spec_seed,
)


def test_category_counts_match_the_real_product():
    """Soul Cast ships 14 genres and 12 archetypes. Guard against drift."""
    assert len(GENRES) == 14
    assert len(ARCHETYPES) == 12
    assert len(set(GENRES)) == 14
    assert len(set(ARCHETYPES)) == 12


def test_budget_bands():
    assert budget_band(0) == "gritty_indie"
    assert budget_band(33) == "gritty_indie"
    assert budget_band(34) == "streaming"
    assert budget_band(66) == "streaming"
    assert budget_band(67) == "blockbuster"
    assert budget_band(100) == "blockbuster"


def test_default_spec_is_valid():
    spec = CharacterSpec(name="santiago")
    assert spec.digest()
    assert spec.handle == "@santiago"


def test_validation_rejects_unknown_enum_values():
    with pytest.raises(ValueError, match="genre"):
        CharacterSpec(name="x", genre="NotAGenre")
    with pytest.raises(ValueError, match="archetype"):
        CharacterSpec(name="x", archetype="NotAnArchetype")
    with pytest.raises(ValueError, match="era"):
        CharacterSpec(name="x", era="1800s")
    with pytest.raises(ValueError, match="outfit"):
        CharacterSpec(name="x", outfit="Spacesuit")


def test_validation_rejects_out_of_range_budget():
    with pytest.raises(ValueError, match="budget"):
        CharacterSpec(name="x", budget=101)
    with pytest.raises(ValueError, match="budget"):
        CharacterSpec(name="x", budget=-1)


def test_manual_details_requires_text():
    with pytest.raises(ValueError, match="details_manual"):
        CharacterSpec(name="x", details="Manual")
    spec = CharacterSpec(name="x", details="Manual", details_manual="a burn scar")
    assert "a burn scar" in compile_prompt(spec)


def test_handle_is_slugged():
    spec = CharacterSpec(name="Old Man River")
    assert spec.handle == "@oldmanriver"
    assert CharacterSpec(name="Éowyn").handle == "@eowyn"


def test_round_trip_through_dict():
    spec = CharacterSpec(name="santiago", genre="Western", budget=20, archetype="Rebel")
    restored = CharacterSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.digest() == spec.digest()


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown CharacterSpec fields"):
        CharacterSpec.from_dict({"name": "x", "not_a_field": 1})


def test_digest_is_stable_and_content_addressed():
    a = CharacterSpec(name="santiago", genre="Western")
    b = CharacterSpec(name="santiago", genre="Western")
    c = CharacterSpec(name="santiago", genre="Horror")
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_seed_is_deterministic_and_in_range():
    spec = CharacterSpec(name="santiago")
    seed = spec_seed(spec)
    assert seed == spec_seed(CharacterSpec(name="santiago"))
    assert 0 <= seed < 2**31


def test_compile_prompt_is_deterministic_and_mentions_the_identity():
    spec = CharacterSpec(
        name="santiago",
        genre="Western",
        era="1900s",
        archetype="Rebel",
        gender="man",
        age="40s",
        build="Lean",
        eye_color="Green",
        hair_style="Long",
        hair_texture="Wavy",
        hair_color="Auburn",
        outfit="Workwear",
        budget=20,
    )
    first = compile_prompt(spec)
    assert first == compile_prompt(spec)
    assert "1900s" in first
    assert "western" in first
    assert "rebel" in first
    assert "green eyes" in first
    assert "auburn" in first
    assert "workwear" in first
    assert "16mm film grain" in first  # low budget band
    assert first.endswith(".")


def test_compile_prompt_omits_unspecified_identity():
    spec = CharacterSpec(name="x", gender="unspecified", race_ethnicity="unspecified")
    prompt = compile_prompt(spec)
    assert "unspecified" not in prompt


def test_compile_bible_contains_every_section():
    spec = CharacterSpec(name="santiago", archetype="Sage")
    bible = compile_bible(spec)
    for section in ("# santiago", "## Identity prompt", "## Physical", "## Wardrobe", "## Story"):
        assert section in bible
    assert spec.digest() in bible
    assert spec.handle in bible


def test_compile_bible_derives_story_from_archetype():
    hero = compile_bible(CharacterSpec(name="a", archetype="Hero"))
    sage = compile_bible(CharacterSpec(name="b", archetype="Sage"))
    assert "unshakeable resolve" in hero
    assert "sees the pattern" in sage


def test_compile_bible_respects_explicit_story_overrides():
    spec = CharacterSpec(name="x", motivation="to finish the job", fear="silence")
    bible = compile_bible(spec)
    assert "to finish the job" in bible
    assert "silence" in bible


def test_every_genre_and_archetype_compiles():
    """No combination should raise."""
    for genre in GENRES:
        for archetype in ARCHETYPES:
            spec = CharacterSpec(name="x", genre=genre, archetype=archetype)
            assert compile_prompt(spec)
            assert compile_bible(spec)


def test_every_era_and_outfit_compiles():
    for era in ERAS:
        for outfit in OUTFITS:
            spec = CharacterSpec(name="x", era=era, outfit=outfit)
            assert era in compile_prompt(spec)
