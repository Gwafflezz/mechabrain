"""The note model: frontmatter parsing, serialization, identity, atomic writes."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from mechabrain.discovery import VaultPaths
from mechabrain.errors import NoteNotFound, SchemaViolation
from mechabrain.note import (
    Note,
    iter_notes,
    normalize_tags,
    note_id_for,
    parse_frontmatter,
    read_note,
    render_note,
    scan_notes,
    serialize_frontmatter,
    wikilink_for,
    write_atomic,
)


# ── Parsing ─────────────────────────────────────────────────────────
def test_parses_frontmatter_and_body() -> None:
    frontmatter, body = parse_frontmatter("---\ntitle: A\nscope: proj-a\n---\nBody text.\n")
    assert frontmatter == {"title": "A", "scope": "proj-a"}
    assert body.strip() == "Body text."


def test_note_without_frontmatter_is_all_body() -> None:
    frontmatter, body = parse_frontmatter("# Just a heading\n\nText.\n")
    assert frontmatter == {}
    assert body == "# Just a heading\n\nText.\n"


def test_empty_frontmatter_block_yields_empty_mapping() -> None:
    frontmatter, body = parse_frontmatter("---\n---\nBody.\n")
    assert frontmatter == {}
    assert body.strip() == "Body."


def test_frontmatter_of_only_comments_yields_empty_mapping() -> None:
    frontmatter, body = parse_frontmatter("---\n# a comment\n---\nBody.\n")
    assert frontmatter == {}
    assert body.strip() == "Body."


def test_only_the_first_closing_fence_ends_the_frontmatter() -> None:
    """`---` is a legal horizontal rule; it must not confuse the parser."""
    text = "---\ntitle: A\n---\nIntro.\n\n---\n\nAfter a rule.\n"
    frontmatter, body = parse_frontmatter(text)
    assert frontmatter == {"title": "A"}
    assert "---" in body
    assert body.strip().endswith("After a rule.")


def test_unterminated_fence_is_treated_as_body() -> None:
    text = "---\n\nA horizontal rule opened the file.\n"
    frontmatter, body = parse_frontmatter(text)
    assert frontmatter == {}
    assert body == text


def test_thematic_break_of_four_dashes_is_not_a_fence() -> None:
    frontmatter, body = parse_frontmatter("----\ntitle: not frontmatter\n")
    assert frontmatter == {}
    assert body.startswith("----")


def test_crlf_is_normalized() -> None:
    frontmatter, body = parse_frontmatter("---\r\ntitle: A\r\n---\r\nBody.\r\n")
    assert frontmatter == {"title": "A"}
    assert "\r" not in body
    assert body.strip() == "Body."


def test_bom_is_stripped() -> None:
    frontmatter, _ = parse_frontmatter("﻿---\ntitle: A\n---\nBody.\n")
    assert frontmatter == {"title": "A"}


def test_unicode_survives_parsing() -> None:
    frontmatter, body = parse_frontmatter("---\ntitle: Ação e coração\n---\nCafé — naïve 中文\n")
    assert frontmatter["title"] == "Ação e coração"
    assert "中文" in body


def test_key_order_is_preserved() -> None:
    frontmatter, _ = parse_frontmatter("---\nzeta: 1\nalpha: 2\nmiddle: 3\n---\n")
    assert list(frontmatter) == ["zeta", "alpha", "middle"]


def test_yaml_dates_become_date_objects() -> None:
    frontmatter, _ = parse_frontmatter("---\ncreated: 2026-01-15\n---\n")
    assert frontmatter["created"] == date(2026, 1, 15)


def test_invalid_yaml_frontmatter_raises(tmp_path: Path) -> None:
    with pytest.raises(SchemaViolation) as excinfo:
        parse_frontmatter("---\ntitle: [unclosed\n---\nBody.\n", path=tmp_path / "n.md")
    assert "invalid YAML" in excinfo.value.message
    assert "n.md" in excinfo.value.message


def test_non_mapping_frontmatter_raises() -> None:
    with pytest.raises(SchemaViolation, match="must be a mapping"):
        parse_frontmatter("---\n- a\n- b\n---\nBody.\n")


# ── Serialization ───────────────────────────────────────────────────
def test_serialize_preserves_key_order_and_does_not_sort() -> None:
    text = serialize_frontmatter({"title": "A", "agent": "alpha", "created": date(2026, 1, 15)})
    assert text.splitlines()[1:4] == ["title: A", "agent: alpha", "created: 2026-01-15"]


def test_serialize_empty_frontmatter_yields_no_fence() -> None:
    assert serialize_frontmatter({}) == ""


def test_serialize_keeps_unicode_unescaped() -> None:
    assert "Ação" in serialize_frontmatter({"title": "Ação"})


def test_serialize_quotes_wikilinks() -> None:
    """`supersedes: [[x]]` unquoted would parse back as a nested YAML list."""
    text = serialize_frontmatter({"supersedes": "[[2026-01-15_INS_x]]"})
    frontmatter, _ = parse_frontmatter(text + "\n")
    assert frontmatter["supersedes"] == "[[2026-01-15_INS_x]]"


def test_serialize_indents_tag_lists() -> None:
    text = serialize_frontmatter({"tags": ["mem/semantic", "agent/alpha"]})
    assert "  - mem/semantic" in text


def test_render_note_normalizes_trailing_newlines() -> None:
    assert render_note({"title": "A"}, "Body.\n\n\n").endswith("Body.\n")
    assert render_note({}, "Body.") == "Body.\n"


def test_round_trip_is_stable() -> None:
    original = (
        "---\ntitle: Ação\ntags:\n  - mem/semantic\n  - agent/alpha\n"
        "created: 2026-01-15\nsupersedes: '[[old]]'\n---\n\n"
        "Body with a rule.\n\n---\n\nAnd more.\n"
    )
    once = Note.parse(original).to_markdown()
    assert Note.parse(once).to_markdown() == once
    reparsed = Note.parse(once)
    assert reparsed.frontmatter["title"] == "Ação"
    assert reparsed.frontmatter["created"] == date(2026, 1, 15)
    assert reparsed.frontmatter["supersedes"] == "[[old]]"
    assert "And more." in reparsed.body


# ── Identity ────────────────────────────────────────────────────────
def test_note_id_is_the_basename_without_extension() -> None:
    assert note_id_for("mecha-brain/Semantic/2026-01-15_INS_x.md") == "2026-01-15_INS_x"
    assert note_id_for(Path("a/b/PROC_deploy.md")) == "PROC_deploy"


def test_note_id_only_strips_the_markdown_suffix() -> None:
    assert note_id_for("2026-01-15_INS_v1.2.md") == "2026-01-15_INS_v1.2"


def test_note_id_is_stable_across_folders() -> None:
    assert note_id_for("Semantic/x.md") == note_id_for("Episodic/alpha/x.md")


def test_wikilink_wraps_the_id() -> None:
    assert wikilink_for("Semantic/2026-01-15_INS_x.md") == "[[2026-01-15_INS_x]]"
    assert wikilink_for("2026-01-15_INS_x") == "[[2026-01-15_INS_x]]"


def test_note_properties(tmp_path: Path) -> None:
    note = Note.parse("---\ntitle: A fact\n---\nBody.\n", path=tmp_path / "2026-01-15_INS_x.md")
    assert note.note_id == "2026-01-15_INS_x"
    assert note.wikilink == "[[2026-01-15_INS_x]]"
    assert note.title == "A fact"
    assert note.get("missing", "fallback") == "fallback"


def test_title_falls_back_to_the_id(tmp_path: Path) -> None:
    assert Note.parse("Body.\n", path=tmp_path / "PROC_deploy.md").title == "PROC_deploy"


def test_unplaced_note_has_empty_identity() -> None:
    assert Note(frontmatter={"title": "A"}).note_id == ""


# ── Tags ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ([], []),
        (["mem/semantic", "agent/alpha"], ["mem/semantic", "agent/alpha"]),
        ("mem/semantic", ["mem/semantic"]),
        ("mem/semantic agent/alpha", ["mem/semantic", "agent/alpha"]),
        ("mem/semantic, agent/alpha", ["mem/semantic", "agent/alpha"]),
        (["#mem/semantic"], ["mem/semantic"]),
        (["", "  ", "#"], []),
    ],
)
def test_normalize_tags(value: object, expected: list[str]) -> None:
    assert normalize_tags(value) == expected


def test_note_tags_and_has_tag() -> None:
    note = Note.parse("---\ntags: [mem/semantic, agent/alpha]\n---\n")
    assert note.tags == ["mem/semantic", "agent/alpha"]
    assert note.has_tag("mem/semantic")
    assert note.has_tag("#mem/semantic")
    assert not note.has_tag("mem/episodic")


# ── Atomic write (R7.5) ─────────────────────────────────────────────
def test_write_atomic_creates_the_file_and_parents(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "note.md"
    assert write_atomic(target, "content\n") == target
    assert target.read_text(encoding="utf-8") == "content\n"


def test_write_atomic_overwrites_in_place(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    write_atomic(target, "first\n")
    write_atomic(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"


def test_write_atomic_leaves_no_temporary_files(tmp_path: Path) -> None:
    write_atomic(tmp_path / "note.md", "content\n")
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_write_atomic_uses_the_destination_directory_for_its_tempfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7.5: the temp file must share the destination's filesystem, or the
    rename degrades to a non-atomic copy across devices."""
    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(str(src))
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    target = tmp_path / "sub" / "note.md"
    write_atomic(target, "content\n")
    assert Path(seen[0]).parent == target.parent


def test_write_atomic_preserves_the_old_content_when_writing_fails(tmp_path: Path) -> None:
    """A crash mid-write leaves the previous note intact and no debris behind."""
    target = tmp_path / "note.md"
    write_atomic(target, "good\n")

    with pytest.raises(TypeError):
        write_atomic(target, object())  # type: ignore[arg-type]

    assert target.read_text(encoding="utf-8") == "good\n"
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_write_atomic_writes_lf_endings(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    write_atomic(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"


def test_write_atomic_handles_unicode(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    write_atomic(target, "Ação — 中文\n")
    assert target.read_text(encoding="utf-8") == "Ação — 中文\n"


# ── Note IO ─────────────────────────────────────────────────────────
def test_note_write_records_its_path(tmp_path: Path) -> None:
    note = Note(frontmatter={"title": "A"}, body="Body.")
    target = note.write(tmp_path / "2026-01-15_INS_x.md")
    assert note.path == target
    assert note.note_id == "2026-01-15_INS_x"
    assert Note.load(target).frontmatter == {"title": "A"}


def test_note_write_without_a_path_raises() -> None:
    with pytest.raises(ValueError, match="no path"):
        Note(body="Body.").write()


def test_note_write_round_trips_through_disk(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    original = Note(path=target, frontmatter={"title": "Ação", "tags": ["mem/semantic"]}, body="Corpo.")
    original.write()
    loaded = read_note(target)
    assert loaded.frontmatter == original.frontmatter
    assert loaded.body.strip() == "Corpo."


def test_load_missing_note_raises(tmp_path: Path) -> None:
    with pytest.raises(NoteNotFound):
        Note.load(tmp_path / "ghost.md")


def test_load_non_utf8_note_raises(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_bytes(b"---\ntitle: \xff\xfe\n---\n")
    with pytest.raises(SchemaViolation, match="UTF-8"):
        Note.load(target)


def test_consolidation_style_edit_round_trip(tmp_path: Path) -> None:
    """§9.3 archives a note by rewriting one frontmatter key; nothing else moves."""
    target = tmp_path / "note.md"
    write_atomic(target, "---\ntitle: A\nstatus: ativo\nscope: proj-a\n---\nBody.\n")
    note = Note.load(target)
    note.frontmatter["status"] = "arquivado"
    note.write()
    reloaded = Note.load(target)
    assert reloaded.frontmatter["status"] == "arquivado"
    assert list(reloaded.frontmatter) == ["title", "status", "scope"]
    assert reloaded.body.strip() == "Body."


# ── Scanning ────────────────────────────────────────────────────────
def test_iter_notes_walks_recursively_in_sorted_order(
    tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    found = scan_notes(tmp_vault.mecha_brain)
    # The fixture also generates AGENTS.md/schema.md, like the real `init` --
    # drop them so the assertion stays about the sample notes' walk order.
    generated = {tmp_vault.agents_file, tmp_vault.schema_file}
    found = [n for n in found if n.path not in generated]
    assert [n.note_id for n in found] == [n.note_id for n in sample_notes]


def test_iter_notes_can_stay_shallow(tmp_vault: VaultPaths, sample_notes: list[Note]) -> None:
    del sample_notes
    assert scan_notes(tmp_vault.episodic_dir, recursive=False) == []
    assert len(scan_notes(tmp_vault.episodic_dir)) == 1


def test_iter_notes_of_a_missing_folder_is_empty(tmp_path: Path) -> None:
    """`Research/` legitimately does not exist when research_enabled is false."""
    assert list(iter_notes(tmp_path / "ghost")) == []


def test_iter_notes_skips_non_markdown_and_dotfiles(tmp_path: Path) -> None:
    write_atomic(tmp_path / "note.md", "---\ntitle: A\n---\n")
    write_atomic(tmp_path / "data.json", "{}")
    write_atomic(tmp_path / ".hidden.md", "---\ntitle: H\n---\n")
    assert [n.note_id for n in iter_notes(tmp_path)] == ["note"]


def test_scan_notes_reads_frontmatter(tmp_vault: VaultPaths, sample_notes: list[Note]) -> None:
    del sample_notes
    scopes = {n.note_id: n.frontmatter["scope"] for n in scan_notes(tmp_vault.semantic_dir)}
    assert scopes["2026-01-15_INS_global-fact"] == "global"
    assert scopes["2026-01-15_INS_vector-search"] == "proj-a"


# ── Fixture contract (relied on by other modules' tests) ────────────
def test_sample_notes_span_types_scopes_and_status(
    tmp_vault: VaultPaths, sample_notes: list[Note], write_note: Callable[..., Note]
) -> None:
    del write_note
    by_id = {note.note_id: note for note in sample_notes}
    assert len(sample_notes) == 6
    assert by_id["2026-01-15_MEM_session-one"].path == (
        tmp_vault.episodic_for("alpha") / "2026-01-15_MEM_session-one.md"
    )
    assert by_id["2026-01-15_INS_stale-fact"].frontmatter["status"] == "arquivado"
    assert by_id["PROC_deploy-playbook"].path == tmp_vault.procedural_dir / "PROC_deploy-playbook.md"
    assert "Evidence" in by_id["PROC_deploy-playbook"].body
    assert "[[2026-01-15_INS_vector-search]]" in by_id["2026-01-15_RES_link-expansion"].body
    assert {note.frontmatter["scope"] for note in sample_notes} == {"proj-a", "proj-b", "global"}


# ══════════════════════════════════════════════════════════════════════
# Regression: no YAML anchors/aliases in written frontmatter (§6)
# ══════════════════════════════════════════════════════════════════════
def test_serialized_frontmatter_never_emits_yaml_anchors() -> None:
    """Reusing one date object must not produce `&id001` / `*id001` (§6).

    The writer stamps `created`, `modified` and `last_accessed` from the same
    ``date`` value; PyYAML's default dumper aliases repeated objects, which
    diverges from the spec's canonical frontmatter and confuses host-app
    frontmatter parsers.
    """
    from datetime import date

    from mechabrain.note import parse_frontmatter, serialize_frontmatter

    stamp = date(2026, 7, 17)
    text = serialize_frontmatter(
        {"title": "X", "created": stamp, "modified": stamp, "last_accessed": stamp}
    )
    assert "&id" not in text and "*id" not in text, text
    parsed, _ = parse_frontmatter(text + "\ncorpo\n")
    assert parsed["created"] == parsed["modified"] == parsed["last_accessed"] == stamp
