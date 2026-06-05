"""Cover the public encode/decode workflows and their schema-header plumbing.

Encode:  encode (lazy expr) | encode_series (eager) | get_header
Decode:  decode (lazy, header bytes) | decode_peek (lazy, sniffs) | decode_series (eager)
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from pl_row_encode import (
    decode,
    decode_peek,
    decode_series,
    encode,
    encode_series,
    get_header,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# --- encode ---------------------------------------------------------------


def test_encode_series_matches_encode_expr(df):
    via_expr = df.select(tok=encode("a", "b"))["tok"]
    via_series = encode_series(df["a"], df["b"])

    assert via_series.dtype == pl.Binary
    assert_series_equal(via_series.rename("tok"), via_expr)


def test_encode_series_requires_input():
    with pytest.raises(ValueError, match="at least one series"):
        encode_series()


# --- get_header -------------------------------------------------------


def test_get_header_from_token_and_series_agree(df):
    tokens = encode_series(df["a"], df["b"])

    from_series = get_header(tokens)
    from_token = get_header(tokens[0])

    assert from_series == from_token
    assert isinstance(from_series, bytes)


def test_get_header_skips_leading_nulls():
    df = pl.DataFrame({"a": [None, 2], "b": [None, "y"]})
    tokens = encode_series(df["a"], df["b"])
    # first row is fully null but still encodes a token; header must be recoverable
    assert get_header(tokens) == get_header(tokens[1])


def test_get_header_empty_series_raises():
    empty = pl.Series("tok", [], dtype=pl.Binary)
    with pytest.raises(ValueError, match="all-null / empty"):
        get_header(empty)


def test_get_header_short_token_raises():
    with pytest.raises(ValueError, match="too short"):
        get_header(b"\x00")


# --- decode (all three paths land on the same frame) ----------------------


def test_decode_with_explicit_header(df):
    tokens = encode_series(df["a"], df["b"])
    header = get_header(tokens)

    out = (
        pl.DataFrame({"tok": tokens})
        .select(decode("tok", schema_header=header).alias("r"))
        .unnest("r")
    )
    assert_frame_equal(out, df)


def test_decode_series_roundtrip(df):
    tokens = encode_series(df["a"], df["b"])
    out = decode_series(tokens).struct.unnest()
    assert_frame_equal(out, df)


@pytest.mark.parametrize("lazy", [False, True])
def test_decode_peek_roundtrip(df, lazy):
    frame = pl.DataFrame({"tok": encode_series(df["a"], df["b"])})
    if lazy:
        frame = frame.lazy()

    expr = decode_peek(frame, "tok").alias("r")
    out = frame.lazy().select(expr).unnest("r").collect()
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, df)


def test_decode_peek_all_null_raises():
    frame = pl.DataFrame({"tok": pl.Series([None, None], dtype=pl.Binary)})
    with pytest.raises(ValueError, match="no non-null token"):
        decode_peek(frame, "tok")


# --- the header is portable across the encode/decode boundary -------------


def test_header_threaded_out_of_band(df):
    # Capture the header once at encode time, then decode lazily elsewhere with only the
    # opaque tokens + the saved header -- no peeking, no eager collect.
    tokens = encode_series(df["a"], df["b"])
    saved_header = get_header(tokens)

    opaque = pl.Series("tok", tokens.to_list(), dtype=pl.Binary)
    out = (
        pl.LazyFrame({"tok": opaque})
        .select(decode("tok", schema_header=saved_header).alias("r"))
        .unnest("r")
        .collect()
    )
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, df)
