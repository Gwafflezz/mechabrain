"""Exception hierarchy.

Two conventions every kernel module follows (R5.1 -- fail loud, no silent
defaults):

1. **Every error names the spec rule it enforces.** ``rule="R6.1"`` renders as
   a ``[R6.1]`` prefix, so a user can grep the spec for the clause that
   rejected them instead of guessing at intent.
2. **Every error is actionable.** ``hint=`` carries the next command to run or
   the exact key to fix -- never a restatement of the message.

Catch :class:`MechabrainError` to catch anything the kernel raises on purpose.
"""

from __future__ import annotations

__all__ = [
    "MechabrainError",
    "ManifestError",
    "VaultNotFoundError",
    "KernelTooOldError",
    "SchemaViolation",
    "DenylistViolation",
    "GateRejected",
    "NoteNotFound",
    "MechabrainIndexError",
    "IndexError",
]


class MechabrainError(Exception):
    """Base of every error the kernel raises deliberately.

    Args:
        message: What went wrong, in the imperative-free indicative.
        rule: Spec rule violated, e.g. ``"R6.1"``. Subclasses set a
            ``default_rule`` so callers may omit it.
        hint: What the caller should *do*. Rendered on its own line.
    """

    #: Rule used when a raiser passes none. Overridden per subclass.
    default_rule: str | None = None

    def __init__(
        self,
        message: str,
        *,
        rule: str | None = None,
        hint: str | None = None,
    ) -> None:
        self.message = message
        self.rule = rule if rule is not None else self.default_rule
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        text = f"[{self.rule}] {self.message}" if self.rule else self.message
        if self.hint:
            text = f"{text}\n  hint: {self.hint}"
        return text

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.message!r}, "
            f"rule={self.rule!r}, hint={self.hint!r})"
        )


class ManifestError(MechabrainError):
    """`_meta/config.yaml` is unreadable, malformed, or violates §5.

    Raised for unknown keys, wrong types, out-of-range values and broken
    cross-references. Carries the dotted path of the offending key in the
    message and, when a near miss exists, the suggested spelling.
    """

    default_rule = "R5.1"


class VaultNotFoundError(MechabrainError):
    """No vault resolved through the R4.3 discovery chain.

    Raised by :func:`mechabrain.discovery.discover_vault` when the explicit
    argument, ``MECHABRAIN_VAULT`` and the upward walk from CWD all miss.
    """

    default_rule = "R4.3"


class KernelTooOldError(MechabrainError):
    """The deployment declares a `kernel_min_version` newer than this kernel.

    The kernel refuses to serve a deployment it does not fully understand
    rather than degrade silently.
    """

    default_rule = "R4.5"


class SchemaViolation(MechabrainError):
    """A note's frontmatter breaks the §6 schema.

    Missing required key, unknown `agent:`, `profile:` absent from the author's
    registry entry, `scope:` outside `scopes.known`, bad `status:`/`confidence:`.
    """

    default_rule = "R6.1"


class DenylistViolation(SchemaViolation):
    """A note carries a key from `denylist_keys` or a tag from `denylist_tags`.

    A :class:`SchemaViolation` subclass: the denylist is part of the schema
    contract, so ``except SchemaViolation`` catches it too.
    """

    default_rule = "R6.1"


class GateRejected(MechabrainError):
    """The §8.2 write gate refused a `memory_write`.

    Only the mechanically checkable items reject here (duplicate in scope,
    missing `source:`, invalid scope, procedural without evidence, denylists).
    Judgement items -- "is this reusable?", "is this atomic?" -- are the agent's
    call and surface as warnings, never as this exception.

    Args:
        near_duplicates: Wikilinks of same-scope notes above `dedup_similarity`,
            for the caller to `supersedes`, merge, or drop (§8.2 item 2).
    """

    default_rule = "R8.2"

    def __init__(
        self,
        message: str,
        *,
        rule: str | None = None,
        hint: str | None = None,
        near_duplicates: list[str] | None = None,
    ) -> None:
        super().__init__(message, rule=rule, hint=hint)
        self.near_duplicates = near_duplicates or []


class NoteNotFound(MechabrainError):
    """No note matches the given id, wikilink or path."""

    default_rule = "R7.1"


class MechabrainIndexError(MechabrainError):
    """The derived index is missing, stale, locked or corrupt.

    The index is always rebuildable (P1), so the hint should point at
    ``mechabrain reindex --full``.

    Exported as both ``MechabrainIndexError`` and ``IndexError``. The spec
    names it ``IndexError``; that alias shadows the builtin when
    star-imported, so kernel modules import the prefixed name.
    """

    default_rule = "R7.4"


#: Spec-facing alias. Shadows the builtin -- import deliberately, or prefer
#: :class:`MechabrainIndexError`.
IndexError = MechabrainIndexError  # noqa: A001
