from datetime import datetime, timezone

import polars as pl
from polars.testing import assert_frame_equal

from pl_row_encode import decode_series, encode


def test_int_and_str_roundtrip():
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "carol"]})

    tokens = df.select(tok=encode("id", "name"))["tok"]
    assert tokens.dtype == pl.Binary

    decoded = decode_series(tokens).struct.unnest()
    assert_frame_equal(decoded, df)


def test_timestamp_tz_roundtrip():
    df = pl.DataFrame(
        {
            "ts": [
                datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
                datetime(2025, 6, 2, 13, 30, tzinfo=timezone.utc),
            ]
        },
        schema={"ts": pl.Datetime("ns", "UTC")},
    )

    tokens = df.select(tok=encode("ts"))["tok"]
    decoded = decode_series(tokens).struct.unnest()

    assert decoded.schema["ts"] == pl.Datetime("ns", "UTC")
    assert_frame_equal(decoded, df)


def test_nulls_roundtrip():
    df = pl.DataFrame({"id": [1, None, 3], "name": [None, "bob", None]})

    tokens = df.select(tok=encode("id", "name"))["tok"]
    decoded = decode_series(tokens).struct.unnest()

    assert_frame_equal(decoded, df)


def test_self_describing_no_external_schema():
    # Encode, then forget everything about the source columns. Decoding works purely from
    # the opaque Binary token, proving the schema is embedded in the bytes.
    df = pl.DataFrame({"a": [10, 20], "b": ["x", "y"]})
    tokens = df.select(tok=encode("a", "b"))["tok"]

    opaque = pl.Series("tok", tokens.to_list(), dtype=pl.Binary)
    decoded = decode_series(opaque).struct.unnest()

    assert decoded.columns == ["a", "b"]
    assert_frame_equal(decoded, df)


def test_token_is_opaque_binary():
    df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    tokens = df.select(tok=encode("id", "name"))["tok"]

    assert tokens.dtype == pl.Binary
    # every row encodes to non-empty bytes
    assert all(len(v) > 0 for v in tokens)
