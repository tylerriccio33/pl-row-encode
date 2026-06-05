"""Row-level, type-preserving encode/decode for Polars columns.

`encode(*cols)` packs a set of columns into a single `Binary` column where each value is
an opaque, self-describing token (the polars-row encoding of the row plus an embedded
schema header). `decode(...)` / `decode_series(...)` reverse it back into a `Struct`.

The token is self-describing, so the schema does not need to be stored anywhere external
to round-trip through a vendor that only holds the opaque bytes.

Encode paths:
    * :func:`encode` -- lazy expr producing a token column.
    * :func:`encode_series` -- eager, returns the token `Series` directly.
    * :func:`get_header` -- pull the schema header out of a token / token `Series`.

Decode paths:
    * :func:`decode` -- fully lazy; you supply the ``schema_header`` bytes (zero peeking).
    * :func:`decode_peek` -- lazy bulk decode; sniffs the header from one materialized token.
    * :func:`decode_series` -- fully eager, schema-free.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExpr
    from typing import Union

    Frame = Union[pl.DataFrame, pl.LazyFrame]

__all__ = [
    "encode",
    "encode_series",
    "get_header",
    "decode",
    "decode_peek",
    "decode_series",
]

_LIB = Path(__file__).parent

# TODO: Doctests for all


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


def encode_series(*series: pl.Series) -> pl.Series:
    """Eagerly encode one or more `Series` into a single `Binary` token `Series`.

    The eager counterpart to :func:`encode`: instead of a lazy expression it returns the
    materialized token column directly, so callers holding `Series` (not a frame) can get
    tokens — and, via :func:`get_header`, the schema header — in one step.
    """
    if not series:
        msg = "encode_series() requires at least one series"
        raise ValueError(msg)
    return pl.select(encode(*(pl.lit(s) for s in series))).to_series()


def get_header(token: bytes | pl.Series) -> bytes:
    """Lift the `[u32 len][header]` schema prefix out of a token.

    Accepts either a single token (`bytes`) or a whole token `Series`, in which case the
    header is read from its first non-null value. The returned bytes are exactly what
    :func:`decode` expects as its ``schema_header`` argument, enabling a fully lazy decode
    without re-sniffing the data.
    """
    if isinstance(token, pl.Series):
        first = next((v for v in token if v is not None), None)
        if first is None:
            msg = "cannot extract header from an all-null / empty Series"
            raise ValueError(msg)
        token = first
    if len(token) < 4:
        msg = "token too short to contain a schema header"
        raise ValueError(msg)
    (header_len,) = struct.unpack_from("<I", token, 0)
    return token[: 4 + header_len]


# TODO: Should be able to do this without the schema_header
# TODO: What about regular dataframes?
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


def decode_peek(frame: Frame, column: str) -> pl.Expr:
    """Build a lazy `decode` expr for `column`, sniffing the schema header for you.

    Unlike :func:`decode`, the caller does not supply a ``schema_header``. This collects a
    single token from `frame` to read the embedded header, then returns the normal lazy
    `decode` expression with that header baked in -- so the bulk decode still runs lazily
    while only one value is materialized up front.

    `frame` may be a `DataFrame` or `LazyFrame`. A peek is required because the output
    `Struct` dtype must be known before the lazy engine sees any data (see :func:`decode`).
    """
    peek = cast(
        "pl.DataFrame",
        frame.lazy().select(pl.col(column).drop_nulls().first()).collect(),
    )
    first = peek.to_series()[0] if peek.height else None
    if first is None:
        msg = (
            "cannot infer schema: no non-null token in the column; "
            "use decode(schema_header=...)"
        )
        raise ValueError(msg)
    header = get_header(first)
    return decode(pl.col(column), schema_header=header)


# TODO: Should try and unify this with the regular decoder
def decode_series(s: pl.Series) -> pl.Series:
    """Eagerly decode a `Binary` token Series into a `Struct` Series, schema-free.

    The schema header is read directly from the first non-null token, so the caller does
    not need to supply or retain any schema.
    """
    header = get_header(s)
    return pl.select(decode(pl.lit(s), schema_header=header)).to_series()
