"""Freeze the *behavior* users depend on: round-trips, dtypes, errors, wire format.

APPEND-ONLY within a major version. You may add tests for new, additive behavior; you may
not weaken or delete an assertion here. A red test in this file means a breaking change.
See CONTRACT.md.
"""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from pl_row_encode import (
    decode,
    decode_peek,
    decode_series,
    encode,
    encode_series,
    get_header,
)

# --- round-trip behavior --------------------------------------------------


def test_encode_produces_binary_token():
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    out = df.select(encode("x", "y").alias("tok"))
    assert out.schema == pl.Schema({"tok": pl.Binary})


def test_lazy_roundtrip_via_decode_with_header():
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}).select(encode("x", "y").alias("tok"))
    header = get_header(df.to_series())
    rows = df.select(decode("tok", schema_header=header).alias("row")).to_series().to_list()
    assert rows == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]


def test_lazy_roundtrip_via_decode_peek():
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}).select(encode("x", "y").alias("tok"))
    rows = df.select(decode_peek(df, "tok").alias("row")).to_series().to_list()
    assert rows == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]


def test_eager_roundtrip_via_series():
    tok = encode_series(pl.Series("x", [10, 20]))
    assert tok.dtype == pl.Binary
    assert decode_series(tok).to_list() == [{"x": 10}, {"x": 20}]


def test_encode_series_matches_encode_expr():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    via_expr = df.select(tok=encode("a", "b"))["tok"]
    via_series = encode_series(df["a"], df["b"])
    assert_series_equal(via_series.rename("tok"), via_expr)


# --- documented errors ----------------------------------------------------


def test_encode_requires_input():
    with pytest.raises(ValueError, match="at least one column"):
        encode()


def test_encode_series_requires_input():
    with pytest.raises(ValueError, match="at least one series"):
        encode_series()


def test_get_header_rejects_all_null_series():
    with pytest.raises(ValueError, match="all-null"):
        get_header(pl.Series("x", [None], dtype=pl.Binary))


def test_get_header_rejects_short_token():
    with pytest.raises(ValueError, match="too short"):
        get_header(b"\x00")


def test_decode_peek_rejects_no_non_null_token():
    df = pl.DataFrame({"tok": pl.Series([None], dtype=pl.Binary)})
    with pytest.raises(ValueError, match="no non-null token"):
        decode_peek(df, "tok")


# --- header shape ---------------------------------------------------------


def test_header_is_length_prefixed():
    import struct

    tok = encode_series(pl.Series("x", [1, 2, 3]))
    header = get_header(tok)
    (header_len,) = struct.unpack_from("<I", header, 0)
    assert header_len == len(header) - 4
    # header sniffed from a single token equals header sniffed from the series
    assert get_header(tok[0]) == header


# --- token wire format (cross-version stability) --------------------------

# A token produced by an early release. Any later release in the same major version MUST
# still decode it to the same rows. If this test fails, the on-disk/over-the-wire token
# format changed -- that is a breaking change for anyone who persisted tokens. Bump major.
GOLDEN_TOKEN = bytes.fromhex(
    "220000000200000000000000010000000000000078090000000100000000000000790e000000018000"
    "0000000000010161"
)
GOLDEN_ROWS = [{"x": 1, "y": "a"}]


def test_golden_token_still_decodes():
    decoded = decode_series(pl.Series("tok", [GOLDEN_TOKEN], dtype=pl.Binary)).to_list()
    assert decoded == GOLDEN_ROWS
