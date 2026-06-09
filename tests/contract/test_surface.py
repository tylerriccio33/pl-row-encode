"""Freeze the *shape* of the public API: exported names and exact signatures.

APPEND-ONLY within a major version. A failure here means the public surface drifted.
If the drift is additive (new symbol, new optional kwarg), update the snapshot in this
file and ship a minor release. If it removes or retypes something, it is a breaking
change: bump the major version and record it in CONTRACT.md.

See CONTRACT.md for the full policy.
"""

import inspect

import pl_row_encode as m

# The frozen set of public symbols. Add to this set for additive releases; never remove.
EXPORTED = {
    "encode",
    "encode_series",
    "get_header",
    "decode",
    "decode_peek",
    "decode_series",
}

# Exact string form of each public signature. Adding an *optional* keyword argument is
# additive (update the snapshot, minor release); changing or removing a parameter is
# breaking (major release).
SIGNATURES = {
    "encode": "(*exprs: 'IntoExpr') -> 'pl.Expr'",
    "encode_series": "(*series: 'pl.Series') -> 'pl.Series'",
    "get_header": "(token: 'bytes | pl.Series') -> 'bytes'",
    "decode": "(expr: 'IntoExpr', *, schema_header: 'bytes') -> 'pl.Expr'",
    "decode_peek": "(frame: 'Frame', column: 'str') -> 'pl.Expr'",
    "decode_series": "(s: 'pl.Series') -> 'pl.Series'",
}


def test_all_matches_frozen_set():
    assert set(m.__all__) == EXPORTED


def test_exported_symbols_are_importable():
    for name in EXPORTED:
        assert hasattr(m, name), f"{name} is exported in __all__ but not importable"


def test_no_undeclared_signatures():
    # Every exported symbol must have a pinned signature below, so new public API can't
    # slip in without consciously freezing its signature.
    assert set(SIGNATURES) == EXPORTED


def test_signatures_frozen():
    for name, expected in SIGNATURES.items():
        actual = str(inspect.signature(getattr(m, name)))
        assert actual == expected, (
            f"signature of {name} changed: {actual!r} != frozen {expected!r}. "
            "If additive, update SIGNATURES (minor release); if not, bump major."
        )
