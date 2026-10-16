"""Contract tests for the ComfyUI node wrappers.

These run without ComfyUI installed. They assert the contract ComfyUI itself
depends on: INPUT_TYPES shape, declared returns matching what FUNCTION
returns, and the mappings being complete.
"""

from __future__ import annotations

import pytest

from nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def test_mapping_and_display_names_are_aligned():
    assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)


def test_every_node_has_the_required_attributes():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, "INPUT_TYPES"), name
        assert hasattr(cls, "RETURN_TYPES"), name
        assert hasattr(cls, "FUNCTION"), name
        assert hasattr(cls, "CATEGORY"), name
        assert isinstance(cls.RETURN_TYPES, tuple), name
        assert callable(getattr(cls, cls.FUNCTION, None)), (
            f"{name}: FUNCTION {cls.FUNCTION!r} is not a method"
        )


def test_every_node_input_types_is_well_formed():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        spec = cls.INPUT_TYPES()
        assert isinstance(spec, dict), name
        for section in spec:
            assert section in {"required", "optional", "hidden"}, f"{name}: {section}"
        for section, fields in spec.items():
            assert isinstance(fields, dict), f"{name}.{section}"
            for field, decl in fields.items():
                assert isinstance(decl, tuple), f"{name}.{section}.{field}"
                # ComfyUI accepts both ("TYPE",) and ("TYPE", {options}).
                assert len(decl) in (1, 2), f"{name}.{section}.{field}"
                kind = decl[0]
                assert isinstance(kind, (list, str)), f"{name}.{field}"
                if len(decl) == 2:
                    assert isinstance(decl[1], dict), f"{name}.{field}"
                else:
                    assert isinstance(kind, str) and kind.isupper(), (
                        f"{name}.{field}: a bare declaration must be a socket type"
                    )


def test_return_names_match_return_types_length():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        names = getattr(cls, "RETURN_NAMES", None)
        if names is not None:
            assert len(names) == len(cls.RETURN_TYPES), (
                f"{name}: {len(names)} names vs {len(cls.RETURN_TYPES)} types"
            )


def test_categories_are_namespaced():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        assert cls.CATEGORY.startswith("SoulFoundry/"), name


def test_every_node_has_a_description():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        assert getattr(cls, "DESCRIPTION", "").strip(), name


# --- behaviour --------------------------------------------------------------


def test_cast_param_grid_emits_a_spec():
    from nodes.cast_param_grid import CastParamGrid

    spec = CastParamGrid().build(
        name="santiago", genre="Western", budget=20, era="1900s",
        archetype="Rebel", gender="man", race_ethnicity="unspecified",
        age="40s", build="Lean", height="tall", eye_color="Green",
        hair_style="Long", hair_texture="Wavy", hair_color="Auburn",
        facial_hair="Stubble", details="Scars", outfit="Workwear",
    )[0]
    assert spec["name"] == "santiago"
    assert spec["genre"] == "Western"
    assert spec["budget"] == 20


def test_sheet_builder_returns_prompt_identity_and_seed():
    from nodes.cast_param_grid import CastParamGrid
    from nodes.sheet_builder import SheetBuilder

    spec = CastParamGrid().build(
        name="santiago", genre="Drama", budget=50, era="Contemporary",
        archetype="Everyman", gender="unspecified", race_ethnicity="unspecified",
        age="30s", build="Average", height="average height", eye_color="Brown",
        hair_style="Short", hair_texture="Straight", hair_color="Dark Brown",
        facial_hair="None", details="None", outfit="Casual",
    )[0]
    prompt, identity, seed = SheetBuilder().build(spec)
    assert "5600K" in prompt
    assert identity
    assert isinstance(seed, int)


def test_sheet_builder_is_deterministic():
    from nodes.cast_param_grid import CastParamGrid
    from nodes.sheet_builder import SheetBuilder

    args = dict(
        name="santiago", genre="Drama", budget=50, era="Contemporary",
        archetype="Everyman", gender="unspecified", race_ethnicity="unspecified",
        age="30s", build="Average", height="average height", eye_color="Brown",
        hair_style="Short", hair_texture="Straight", hair_color="Dark Brown",
        facial_hair="None", details="None", outfit="Casual",
    )
    grid, builder = CastParamGrid(), SheetBuilder()
    assert builder.build(grid.build(**args)[0]) == builder.build(grid.build(**args)[0])


def test_sheet_qc_metrics_mode():
    from nodes.sheet_qc import SheetQC

    report, passed, score, reasons = SheetQC().run(
        mode="metrics",
        metrics_json=(
            '{"panels":3,"shadow_ratio":0.2,"highlight_clip":0.005,'
            '"catchlight":0.4,"backdrop_delta":4.0,"subject_coverage":0.3}'
        ),
    )
    assert passed is True
    assert score == 1.0
    assert reasons == ""
    assert report["passed"] is True


def test_sheet_qc_image_mode_without_image_raises():
    from nodes.sheet_qc import SheetQC

    with pytest.raises(ValueError, match="requires an IMAGE"):
        SheetQC().run(mode="image")


def test_sheet_qc_bad_json_raises():
    from nodes.sheet_qc import SheetQC

    with pytest.raises(ValueError, match="not valid JSON"):
        SheetQC().run(mode="metrics", metrics_json="{oops")


def test_sheet_qc_batch_picks_the_best():
    from nodes.sheet_qc import SheetQCBatch

    reports = (
        '[{"passed":false,"score":0.4},{"passed":true,"score":0.9},'
        '{"passed":false,"score":0.5}]'
    )
    best, report = SheetQCBatch().run(reports)
    assert best == 1
    assert report["score"] == 0.9


def test_mention_resolver_expands():
    from nodes.mention_resolver import MentionResolver

    expanded, mentions, count = MentionResolver().resolve(
        "@santiago walks in",
        library_json='{"santiago": "Tall, grey eyes."}',
    )
    assert "Tall, grey eyes." in expanded
    assert mentions == "@santiago"
    assert count == 1


def test_element_injector_strict_raises_on_unknown():
    from nodes.element_injector import ElementInjector

    with pytest.raises(ValueError, match="not in the library"):
        ElementInjector().inject("@nobody", library_json="{}")


def test_element_injector_passes_through_when_no_mentions():
    from nodes.element_injector import ElementInjector

    out, report = ElementInjector().inject("a plain shot prompt")
    assert out == "a plain shot prompt"
    assert "no @mentions" in report


def test_location_and_prop_elements_build():
    from nodes.element_injector import LocationElement, PropElement

    loc = LocationElement().build("docks", "Rain-slick docks at night")[0]
    prop = PropElement().build("revolver", "Worn navy revolver")[0]
    assert loc["kind"] == "location"
    assert prop["kind"] == "prop"


def test_consistency_audit_scores_mode():
    from nodes.consistency_audit import ConsistencyAudit

    report, mean, holding, md = ConsistencyAudit().audit(
        mode="scores", scores_json="[0.95,0.93,0.91]",
    )
    assert holding is True
    assert mean == pytest.approx(0.93)
    assert "Drift report" in md


def test_consistency_audit_embeddings_mode():
    from nodes.consistency_audit import ConsistencyAudit

    report, mean, holding, _ = ConsistencyAudit().audit(
        mode="embeddings",
        reference_json="[1.0, 0.0]",
        frames_json="[[1.0,0.0],[0.9,0.1]]",
    )
    assert report["shots"] == 2
    assert mean > 0.9


def test_consistency_audit_rejects_mismatched_vector_lengths():
    from nodes.consistency_audit import ConsistencyAudit

    with pytest.raises(ValueError, match="length mismatch"):
        ConsistencyAudit().audit(
            mode="embeddings",
            reference_json="[1.0, 0.0]",
            frames_json="[[1.0,0.0,0.0]]",
        )


def test_shot_list_runner_resolves_every_shot():
    from nodes.shot_list_runner import ShotListRunner

    plan, report = ShotListRunner().run(
        shots_json='[{"seq":1,"prompt":"@santiago enters"},'
                  '{"seq":2,"prompt":"@santiago leaves"}]',
        library_json='{"santiago": "Tall, grey eyes."}',
    )
    assert plan["count"] == 2
    assert all("Tall, grey eyes." in s["prompt"] for s in plan["shots"])
    assert "@santiago (2x)" in report


def test_shot_list_runner_accepts_bare_strings():
    from nodes.shot_list_runner import ShotListRunner

    plan, _ = ShotListRunner().run(shots_json='["a","b","c"]')
    assert [s["seq"] for s in plan["shots"]] == [1, 2, 3]


def test_cast_sheet_export_markdown_and_json():
    from nodes.cast_param_grid import CastParamGrid
    from nodes.cast_sheet_export import CastSheetExport

    spec = CastParamGrid().build(
        name="santiago", genre="Drama", budget=50, era="Contemporary",
        archetype="Everyman", gender="unspecified", race_ethnicity="unspecified",
        age="30s", build="Average", height="average height", eye_color="Brown",
        hair_style="Short", hair_texture="Straight", hair_color="Dark Brown",
        facial_hair="None", details="None", outfit="Casual",
    )[0]
    md, name = CastSheetExport().export(format="markdown", spec=spec)
    assert "# santiago" in md
    assert name == "santiago-bible.md"

    payload, jname = CastSheetExport().export(format="json", spec=spec)
    assert '"name": "santiago"' in payload
    assert jname == "santiago.json"


def test_cast_sheet_export_requires_a_spec_or_element():
    from nodes.cast_sheet_export import CastSheetExport

    with pytest.raises(ValueError, match="at least one of spec or element"):
        CastSheetExport().export(format="markdown")


def test_character_lock_requires_the_right_artifact_per_route():
    from nodes.cast_param_grid import CastParamGrid
    from nodes.character_lock import CharacterLock

    spec = CastParamGrid().build(
        name="santiago", genre="Drama", budget=50, era="Contemporary",
        archetype="Everyman", gender="unspecified", race_ethnicity="unspecified",
        age="30s", build="Average", height="average height", eye_color="Brown",
        hair_style="Short", hair_texture="Straight", hair_color="Dark Brown",
        facial_hair="None", details="None", outfit="Casual",
    )[0]
    lock = CharacterLock()
    with pytest.raises(ValueError, match="needs an embed_path"):
        lock.lock(spec, route="embedding")
    with pytest.raises(ValueError, match="needs lora_path"):
        lock.lock(spec, route="lora")

    element, status = lock.lock(spec, route="embedding", embed_path="/tmp/e.pt")
    assert element["name"] == "santiago"
    assert element["spec_digest"]
    assert "embedding" in status


def test_character_lock_persists_and_pins(tmp_path):
    from nodes.cast_param_grid import CastParamGrid
    from nodes.character_lock import CharacterLock
    from nodes.core.store import ProjectStore

    spec = CastParamGrid().build(
        name="santiago", genre="Drama", budget=50, era="Contemporary",
        archetype="Everyman", gender="unspecified", race_ethnicity="unspecified",
        age="30s", build="Average", height="average height", eye_color="Brown",
        hair_style="Short", hair_texture="Straight", hair_color="Dark Brown",
        facial_hair="None", details="None", outfit="Casual",
    )[0]
    CharacterLock().lock(
        spec, route="embedding", embed_path="/tmp/e.pt",
        bible_md="bible", project_path=str(tmp_path / "proj"),
    )
    with ProjectStore(tmp_path / "proj") as store:
        saved = store.get_element("santiago")
        assert saved is not None
        assert saved.pinned is True


def test_element_library_list_and_load(tmp_path):
    from nodes.core.store import Element, ProjectStore
    from nodes.element_library import ElementLibrary

    path = str(tmp_path / "proj")
    with ProjectStore(path) as store:
        store.save_element(Element(name="santiago", bible_md="bible"))

    lib = ElementLibrary()
    element, report, count = lib.run(path, action="list")
    assert count == 1
    assert "@santiago" in report

    element, report, count = lib.run(path, action="load", name="santiago")
    assert element["bible_md"] == "bible"


def test_element_library_save_and_delete(tmp_path):
    from nodes.element_library import ElementLibrary

    path = str(tmp_path / "proj")
    lib = ElementLibrary()
    payload = {"name": "marisol", "kind": "character", "bible_md": "text"}
    _, report, count = lib.run(path, action="save", element=payload)
    assert count == 1
    assert "@marisol" in report

    _, report, count = lib.run(path, action="delete", name="marisol")
    assert count == 1


def test_element_library_load_missing_raises(tmp_path):
    from nodes.element_library import ElementLibrary

    with pytest.raises(KeyError):
        ElementLibrary().run(str(tmp_path / "proj"), action="load", name="nobody")
