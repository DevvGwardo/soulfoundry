"""Tests for mention resolution and expansion."""

from __future__ import annotations

import pytest

from nodes.core.mentions import (
    MentionError,
    expand_mentions,
    find_mentions,
    library_from_specs,
)

LIB = {
    "santiago": "# santiago\n\nDark hair, grey eyes, workwear.",
    "marisol": "# marisol\n\nRed hair, green eyes, formal.",
}


def test_find_mentions_returns_distinct_in_order():
    assert find_mentions("@santiago meets @marisol, then @santiago again") == [
        "santiago",
        "marisol",
    ]


def test_find_mentions_is_case_insensitive_and_dedupes():
    assert find_mentions("@Santiago and @santiago") == ["santiago"]


def test_find_mentions_ignores_email_like_text():
    assert find_mentions("write to bob@example.com") == []


def test_find_mentions_does_not_half_match_longer_handle():
    """@santiago2 must not register as @santiago."""
    assert find_mentions("@santiago2 walks in") == ["santiago2"]


def test_find_mentions_handles_empty_input():
    assert find_mentions("") == []
    assert find_mentions(None) == []


def test_find_mentions_accepts_hyphens_and_underscores():
    assert find_mentions("@old_man @river-run") == ["old_man", "river-run"]


def test_inline_expansion_substitutes_verbatim_bible():
    out = expand_mentions("@santiago walks in", LIB)
    assert "Dark hair, grey eyes, workwear." in out
    assert "@santiago" not in out


def test_inline_expansion_flattens_newlines():
    out = expand_mentions("@santiago", LIB)
    assert "\n" not in out


def test_inline_expansion_truncates_with_ellipsis():
    lib = {"x": "a" * 5000}
    out = expand_mentions("@x", lib, max_inline_chars=100)
    assert len(out) <= 100
    assert out.endswith("…")


def test_append_style_keeps_handle_and_adds_block():
    out = expand_mentions("@santiago and @marisol", LIB, style="append")
    assert "@santiago" in out
    assert "REFERENCE ELEMENTS" in out
    assert "[@santiago]" in out
    assert "[@marisol]" in out
    # each referenced element appears exactly once in the block
    assert out.count("[@santiago]") == 1


def test_append_style_does_not_duplicate_repeated_mention():
    out = expand_mentions("@santiago and @santiago", LIB, style="append")
    assert out.count("[@santiago]") == 1


def test_strip_style_removes_mentions_entirely():
    out = expand_mentions("@santiago walks in", LIB, style="strip")
    assert "@santiago" not in out
    assert "Dark hair" not in out
    assert "walks in" in out


def test_unknown_mention_raises_in_strict_mode():
    with pytest.raises(MentionError, match="not in the library"):
        expand_mentions("@nobody", LIB)


def test_unknown_mention_is_left_alone_when_not_strict():
    out = expand_mentions("@nobody walks in", LIB, strict=False)
    assert "@nobody" in out


def test_unknown_style_rejected():
    with pytest.raises(ValueError, match="unknown style"):
        expand_mentions("@santiago", LIB, style="shout")


def test_library_keys_may_carry_at_sign():
    out = expand_mentions("@santiago", {"@Santiago": "bible text"})
    assert "bible text" in out


def test_library_from_specs_builds_handles():
    from nodes.core.param_grid import CharacterSpec, compile_bible

    specs = [CharacterSpec(name="Santiago"), CharacterSpec(name="Marisol")]
    lib = library_from_specs(specs, compile_bible)
    assert set(lib) == {"santiago", "marisol"}
    assert "Identity prompt" in lib["santiago"]


def test_expansion_is_idempotent_on_already_expanded_text():
    """Running twice must not double-insert the bible."""
    once = expand_mentions("@santiago", LIB)
    twice = expand_mentions(once, LIB)
    assert once == twice


def test_backticked_handle_is_prose_not_a_mention():
    """The compiled bible documents its own handle as `@santiago`.

    That is text *about* the handle, not a reference to resolve. If it matched,
    expanding a prompt would re-inject a mention from the bible's own header
    and expansion would stop being idempotent.
    """
    assert find_mentions("see `@santiago` in the docs") == []
    assert find_mentions("`@santiago` and @marisol") == ["marisol"]


def test_expanding_a_real_compiled_bible_is_idempotent():
    """The exact failure the end-to-end demo caught."""
    from nodes.core.param_grid import CharacterSpec, compile_bible

    spec = CharacterSpec(name="santiago")
    library = {"santiago": compile_bible(spec)}
    once = expand_mentions("@santiago walks in", library)
    assert "walks in" in once
    assert expand_mentions(once, library) == once


def test_compiled_bible_references_its_own_handle():
    from nodes.core.param_grid import CharacterSpec, compile_bible

    bible = compile_bible(CharacterSpec(name="santiago"))
    assert "`@santiago`" in bible
    assert find_mentions(bible) == []
