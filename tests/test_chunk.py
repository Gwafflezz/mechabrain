"""Tests for chunking and deterministic Contextual Retrieval (§7.1).

The load-bearing property, asserted from several angles: the context prefix
lands on ``embed_text`` and **only** there. If it ever leaks into ``raw_text``,
agents start citing `scope: ... | title: ...` back at the user (R7.1).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mechabrain.index.chunk import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    Chunk,
    chunk_note,
    chunk_notes,
    context_prefix,
    split_sections,
)
from mechabrain.manifest import Manifest
from mechabrain.note import Note

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
ATOMIC_NOTE = """---
title: Vector search
tags: [mem/semantic, agent/alpha]
scope: proj-a
---

Brute-force cosine is fast enough below ten thousand chunks.
"""


def make_note(body: str, note_id: str = "2026-01-15_INS_vector-search") -> Note:
    """A note with §6 frontmatter and ``body`` as its Markdown."""
    return Note(
        path=Path(f"{note_id}.md"),
        frontmatter={
            "title": "Vector search",
            "tags": ["mem/semantic", "agent/alpha"],
            "scope": "proj-a",
        },
        body=body,
    )


# ══════════════════════════════════════════════════════════════════════
# Chunk
# ══════════════════════════════════════════════════════════════════════
def test_chunk_id_joins_note_and_ordinal() -> None:
    chunk = Chunk(note_id="PROC_deploy", ordinal=2, raw_text="x", embed_text="x")
    assert chunk.chunk_id == "PROC_deploy#2"


def test_section_renders_heading_path() -> None:
    chunk = Chunk(
        note_id="n", ordinal=0, raw_text="x", embed_text="x",
        heading_path=("Deploy", "Rollback"),
    )
    assert chunk.section == "Deploy > Rollback"


def test_section_is_empty_at_note_level() -> None:
    assert Chunk(note_id="n", ordinal=0, raw_text="x", embed_text="x").section == ""


def test_chunk_is_frozen() -> None:
    """Derived state: re-chunk to fix it, never mutate behind the index."""
    chunk = Chunk(note_id="n", ordinal=0, raw_text="x", embed_text="x")
    with pytest.raises(AttributeError):
        chunk.raw_text = "mutated"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════
# The contextual prefix: embed_text only
# ══════════════════════════════════════════════════════════════════════
def test_prefix_changes_embed_text_and_not_raw_text() -> None:
    """The core §7.1 contract, stated as a differential."""
    note = Note.parse(ATOMIC_NOTE, path="2026-01-15_INS_vector-search.md")

    with_prefix = chunk_note(note, contextual=True)
    without_prefix = chunk_note(note, contextual=False)

    assert len(with_prefix) == len(without_prefix) == 1
    # raw_text is identical either way: the switch does not touch it.
    assert with_prefix[0].raw_text == without_prefix[0].raw_text
    # embed_text is not.
    assert with_prefix[0].embed_text != without_prefix[0].embed_text
    assert with_prefix[0].embed_text.endswith(with_prefix[0].raw_text)


def test_contextual_off_makes_embed_text_equal_raw_text() -> None:
    note = Note.parse(ATOMIC_NOTE, path="2026-01-15_INS_vector-search.md")
    for chunk in chunk_note(note, contextual=False):
        assert chunk.embed_text == chunk.raw_text


def test_prefix_carries_scope_title_and_tags() -> None:
    note = Note.parse(ATOMIC_NOTE, path="2026-01-15_INS_vector-search.md")
    (chunk,) = chunk_note(note, contextual=True)

    prefix = chunk.embed_text.removesuffix(chunk.raw_text)
    assert "scope: proj-a" in prefix
    assert "title: Vector search" in prefix
    assert "tags: mem/semantic, agent/alpha" in prefix
    # None of it contaminates what a human is shown.
    for field in ("scope:", "title:", "tags:"):
        assert field not in chunk.raw_text


@pytest.mark.parametrize("contextual", [True, False])
def test_embed_text_composition_holds_for_every_chunk(contextual: bool) -> None:
    """The invariant both index halves rely on, over a note that actually splits."""
    note = make_note("## A\n\n" + ("alpha " * 400) + "\n\n## B\n\nShort.")
    chunks = chunk_note(note, contextual=contextual, max_chars=500, overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        prefix = context_prefix(note, chunk.heading_path) if contextual else ""
        expected = f"{prefix}\n\n{chunk.raw_text}" if prefix else chunk.raw_text
        assert chunk.embed_text == expected


def test_prefix_carries_heading_path() -> None:
    note = make_note("## Deploy\n\n### Rollback\n\nRevert the tag.")
    (chunk,) = chunk_note(note, contextual=True)
    assert "section: Deploy > Rollback" in chunk.embed_text
    assert "section:" not in chunk.raw_text


def test_context_prefix_omits_absent_fields() -> None:
    """Absent frontmatter is omitted, never filled with a placeholder."""
    note = Note(path=Path("bare.md"), frontmatter={}, body="Body.")
    assert context_prefix(note, ()) == "title: bare"

    scoped = Note(path=Path("bare.md"), frontmatter={"scope": "global"}, body="Body.")
    assert context_prefix(scoped, ()) == "scope: global | title: bare"


def test_no_frontmatter_means_no_prefix_applied() -> None:
    """An unplaced note with no frontmatter has nothing to say about itself."""
    note = Note(path=None, frontmatter={}, body="Just prose.")
    assert context_prefix(note, ()) == ""
    (chunk,) = chunk_note(note, contextual=True)
    assert chunk.embed_text == chunk.raw_text == "Just prose."


def test_prefix_is_deterministic() -> None:
    """No LLM, no clock, no I/O: same bytes in, same bytes out (§13)."""
    note = Note.parse(ATOMIC_NOTE, path="2026-01-15_INS_vector-search.md")
    assert chunk_note(note) == chunk_note(note)


def test_frontmatter_never_lands_in_a_chunk_body() -> None:
    note = Note.parse(ATOMIC_NOTE, path="2026-01-15_INS_vector-search.md")
    for chunk in chunk_note(note, contextual=True):
        assert "---" not in chunk.raw_text
        assert "---" not in chunk.embed_text


# ══════════════════════════════════════════════════════════════════════
# split_sections
# ══════════════════════════════════════════════════════════════════════
def test_split_sections_tracks_nesting() -> None:
    sections = split_sections(
        "Preamble.\n\n# Top\n\nOne.\n\n## Mid\n\nTwo.\n\n### Deep\n\nThree.\n\n## Other\n\nFour."
    )
    assert [path for path, _ in sections] == [
        (),
        ("Top",),
        ("Top", "Mid"),
        ("Top", "Mid", "Deep"),
        ("Top", "Other"),
    ]


def test_split_sections_keeps_the_heading_line_in_the_text() -> None:
    ((_, text),) = split_sections("## Steps\n\nBuild it.")
    assert text.startswith("## Steps")
    assert "Build it." in text


def test_split_sections_drops_empty_sections_but_keeps_their_title_in_the_path() -> None:
    sections = split_sections("## Empty\n\n### Full\n\nContent.")
    assert len(sections) == 1
    assert sections[0][0] == ("Empty", "Full")


def test_split_sections_of_a_blank_body_is_empty() -> None:
    assert split_sections("") == []
    assert split_sections("\n\n   \n") == []


def test_heading_closing_sequence_is_decoration() -> None:
    ((path, _),) = split_sections("## Steps ##\n\nBuild it.")
    assert path == ("Steps",)


def test_hash_tag_is_not_a_heading() -> None:
    """`#tag` has no space after the hash -- vaults are full of these."""
    sections = split_sections("#project/alpha\n\nSome prose.")
    assert [path for path, _ in sections] == [()]


def test_heading_inside_a_code_fence_is_not_a_heading() -> None:
    sections = split_sections("Intro.\n\n```bash\n# rm -rf /\n```\n\n## Real\n\nBody.")
    assert [path for path, _ in sections] == [(), ("Real",)]


def test_unterminated_fence_swallows_the_rest() -> None:
    """CommonMark behaviour -- and it keeps a typo from inventing headings."""
    sections = split_sections("Intro.\n\n```\n## not a heading\n")
    assert [path for path, _ in sections] == [()]


# ══════════════════════════════════════════════════════════════════════
# Budget and merging
# ══════════════════════════════════════════════════════════════════════
def test_atomic_note_is_one_chunk() -> None:
    note = Note.parse(ATOMIC_NOTE, path="2026-01-15_INS_vector-search.md")
    assert len(chunk_note(note)) == 1


def test_short_sibling_sections_merge_into_one_chunk() -> None:
    """A 60-char note with two headings is one thought, not two."""
    note = make_note("## Steps\n\n1. Build.\n2. Ship.\n\n## Evidence\n\nRan on 2026-01-14.")
    chunks = chunk_note(note)
    assert len(chunks) == 1
    # Merged siblings have no single enclosing heading...
    assert chunks[0].heading_path == ()
    # ...but nothing is lost: the heading lines are still in the text.
    assert "## Steps" in chunks[0].raw_text
    assert "## Evidence" in chunks[0].raw_text


def test_a_single_section_keeps_its_own_heading_path() -> None:
    note = make_note("## Steps\n\n1. Build.")
    (chunk,) = chunk_note(note)
    assert chunk.heading_path == ("Steps",)


def test_merged_run_keeps_the_common_ancestor() -> None:
    note = make_note("# Deploy\n\n## Blue\n\nOne.\n\n## Green\n\nTwo.")
    (chunk,) = chunk_note(note)
    assert chunk.heading_path == ("Deploy",)


def test_long_note_splits_and_respects_the_budget() -> None:
    note = make_note("## A\n\n" + ("alpha " * 400) + "\n\n## B\n\n" + ("beta " * 400))
    chunks = chunk_note(note, max_chars=500, overlap=50)

    assert len(chunks) > 2
    for chunk in chunks:
        # The budget bounds consumed body text; overlap and the section heading
        # are replayed on top of it.
        assert len(chunk.raw_text) <= 500 + 50 + len("## A")


def test_a_heading_is_never_a_chunk_of_its_own() -> None:
    """Regression: `## A` used to flush as a 4-char chunk before its prose."""
    note = make_note("## A\n\n" + ("alpha " * 400))
    for chunk in chunk_note(note, max_chars=500, overlap=50):
        assert chunk.raw_text.strip() not in {"## A", "#", ""}
        assert len(chunk.raw_text) > len("## A")


def test_split_pieces_keep_their_section_path() -> None:
    note = make_note("## A\n\n" + ("alpha " * 400))
    chunks = chunk_note(note, max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.heading_path == ("A",) for chunk in chunks)


def test_overlap_replays_the_previous_tail() -> None:
    note = make_note("## A\n\n" + " ".join(f"w{i}" for i in range(400)))
    first, second = chunk_note(note, max_chars=500, overlap=60)[:2]
    tail = first.raw_text[-40:].strip()
    assert tail in second.raw_text


def test_zero_overlap_replays_nothing() -> None:
    note = make_note("## A\n\n" + " ".join(f"w{i}" for i in range(400)))
    first, second = chunk_note(note, max_chars=500, overlap=0)[:2]
    assert not second.raw_text.startswith(first.raw_text[-20:])


def test_overlap_does_not_cross_a_heading() -> None:
    """Across a heading the author declared a topic change; replay would be noise."""
    note = make_note("## A\n\n" + ("alpha " * 200) + "\n\n## B\n\n" + ("beta " * 200))
    chunks = chunk_note(note, max_chars=400, overlap=60)
    for chunk in chunks:
        if chunk.heading_path == ("B",):
            assert "alpha" not in chunk.raw_text


def test_code_fence_survives_paragraph_splitting() -> None:
    """A blank line inside a fence is not a paragraph break."""
    note = make_note("Run it:\n\n```py\nx = 1\n\ny = 2\n```")
    (chunk,) = chunk_note(note)
    assert "x = 1\n\ny = 2" in chunk.raw_text


def test_a_word_longer_than_the_budget_is_hard_split() -> None:
    note = make_note("a" * 900)
    chunks = chunk_note(note, max_chars=300, overlap=0)
    assert len(chunks) == 3
    assert "".join(chunk.raw_text for chunk in chunks) == "a" * 900


def test_empty_body_yields_no_chunks() -> None:
    assert chunk_note(make_note("")) == []
    assert chunk_note(make_note("\n\n  \n")) == []


def test_ordinals_run_from_zero_in_reading_order() -> None:
    note = make_note("## A\n\n" + ("alpha " * 400) + "\n\n## B\n\n" + ("beta " * 400))
    chunks = chunk_note(note, max_chars=400, overlap=40)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.note_id == "2026-01-15_INS_vector-search" for chunk in chunks)


# ══════════════════════════════════════════════════════════════════════
# Argument validation (R5.1)
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    ("max_chars", "overlap", "expected"),
    [
        (0, 0, "max_chars must be positive"),
        (-1, 0, "max_chars must be positive"),
        (100, -1, "overlap must not be negative"),
        (100, 100, "must be smaller than max_chars"),
        (100, 150, "must be smaller than max_chars"),
    ],
)
def test_bad_budget_fails_loudly(max_chars: int, overlap: int, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        chunk_note(make_note("Body."), max_chars=max_chars, overlap=overlap)


# ══════════════════════════════════════════════════════════════════════
# chunk_notes
# ══════════════════════════════════════════════════════════════════════
def test_chunk_notes_concatenates_and_restarts_ordinals(
    sample_notes: list[Note],
) -> None:
    chunks = chunk_notes(sample_notes)

    assert {chunk.note_id for chunk in chunks} == {note.note_id for note in sample_notes}
    # ordinal is per-note; chunk_id is what is unique corpus-wide.
    assert all(chunk.ordinal == 0 for chunk in chunks), "fixtures are atomic notes"
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_notes_of_nothing_is_empty() -> None:
    assert chunk_notes([]) == []


def test_sample_notes_carry_their_scope_into_embed_text(
    sample_notes: list[Note],
) -> None:
    by_id = {note.note_id: note for note in sample_notes}
    (chunk,) = chunk_note(by_id["2026-01-15_INS_stale-fact"], contextual=True)
    assert "scope: proj-b" in chunk.embed_text
    assert "scope: proj-b" not in chunk.raw_text


# ══════════════════════════════════════════════════════════════════════
# Manifest wiring
# ══════════════════════════════════════════════════════════════════════
def test_manifest_drives_the_switch(
    manifest_ci: Manifest, manifest_full: Manifest
) -> None:
    """`contextual_retrieval` is true in the CI manifest, false in the full one."""
    note = Note.parse(ATOMIC_NOTE, path="2026-01-15_INS_vector-search.md")

    (on,) = chunk_note(note, contextual=manifest_ci.retrieval.contextual_retrieval)
    (off,) = chunk_note(note, contextual=manifest_full.retrieval.contextual_retrieval)

    assert on.embed_text.startswith("scope: proj-a")
    assert off.embed_text == off.raw_text


def test_chunk_asdict_feeds_the_lexical_index(sample_notes: list[Note]) -> None:
    """Cross-module contract: BM25 reads the chunker's own spellings (§7.1).

    `lexical` indexes ``embed_text``, so the prefix is findable on the lexical
    side too -- both halves index the same string.
    """
    from dataclasses import asdict

    from mechabrain.index.lexical import LexicalChunk

    note = next(n for n in sample_notes if n.note_id == "2026-01-15_INS_vector-search")
    (chunk,) = chunk_note(note, contextual=True)

    lexical = LexicalChunk.from_mapping(
        {**asdict(chunk), "chunk_id": chunk.chunk_id, "scope": note.get("scope")}
    )
    assert lexical.chunk_id == "2026-01-15_INS_vector-search#0"
    assert lexical.text == chunk.embed_text
    assert "scope: proj-a" in lexical.text
    assert lexical.scope == "proj-a"


def test_defaults_are_the_documented_ones() -> None:
    assert (DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS) == (1200, 120)

    note = make_note("Body.")
    assert chunk_note(note) == chunk_note(
        note, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP_CHARS
    )


def test_chunking_a_note_read_from_disk(
    tmp_vault: Any, write_note: Callable[..., Note]
) -> None:
    """End to end: the frontmatter parsed off disk reaches the prefix."""
    note = write_note(
        tmp_vault.semantic_dir / "2026-01-15_INS_on-disk.md",
        {"title": "On disk", "tags": ["mem/semantic"], "scope": "proj-a"},
        "## Finding\n\nIt round-trips.",
    )
    (chunk,) = chunk_note(Note.load(note.path), contextual=True)

    assert chunk.embed_text == (
        "scope: proj-a | title: On disk | tags: mem/semantic | section: Finding"
        "\n\n## Finding\nIt round-trips."
    )
    assert chunk.raw_text == "## Finding\nIt round-trips."
