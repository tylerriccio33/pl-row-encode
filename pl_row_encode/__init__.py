"""Row-level, type-preserving encode/decode for Polars columns.

`encode(*cols)` packs a set of columns into a single `Binary` column where each value is
an opaque, self-describing token (the polars-row encoding of the row plus an embedded
schema header). `decode(...)` / `decode_series(...)` reverse it back into a `Struct`.

The token is self-describing, so the schema does not need to be stored anywhere external
to round-trip through a vendor that only holds the opaque bytes.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = ["encode", "decode", "decode_series"]

_LIB = Path(__file__).parent


def encode(*exprs: IntoExpr) -> pl.Expr:
    """Encode one or more columns into a single self-describing `Binary` token column."""
    if not exprs:
        msg = "encode() requires at least one column"
        raise ValueError(msg)
    return register_plugin_function(
        plugin_path=_LIB,
        function_name="row_encode",
        args=list(exprs),
        is_elementwise=True,
    )


def _extract_header(token: bytes) -> bytes:
    """Lift the `[u32 len][header]` prefix off a token (the bytes decode needs up front)."""
    if len(token) < 4:
        msg = "token too short to contain a schema header"
        raise ValueError(msg)
    (header_len,) = struct.unpack_from("<I", token, 0)
    return token[: 4 + header_len]


def decode(expr: IntoExpr, *, schema_header: bytes) -> pl.Expr:
    """Decode a `Binary` token column back into a `Struct`.

    `schema_header` is the header prefix of any token in the column (see
    :func:`decode_series` for the eager path that extracts it for you). It is required so
    the output `Struct` dtype can be resolved before the data is materialized, which the
    Polars lazy engine needs.
    """
    return register_plugin_function(
        plugin_path=_LIB,
        function_name="row_decode",
        args=[expr],
        is_elementwise=True,
        kwargs={"schema_header": schema_header},
    )


def decode_series(s: pl.Series) -> pl.Series:
    """Eagerly decode a `Binary` token Series into a `Struct` Series, schema-free.

    The schema header is read directly from the first non-null token, so the caller does
    not need to supply or retain any schema.
    """
    first = next((v for v in s if v is not None), None)
    if first is None:
        msg = "cannot infer schema from an all-null / empty Series; use decode(schema_header=...)"
        raise ValueError(msg)
    header = _extract_header(first)
    return pl.select(decode(pl.lit(s), schema_header=header)).to_series()
