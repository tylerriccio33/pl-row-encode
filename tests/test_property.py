"""Property-based round-trip tests.

Generate arbitrary Polars frames with Hypothesis (via polars' parametric strategies)
and assert that ``encode`` -> ``decode`` is the identity for every public decode path.
If any dtype, null pattern, or value survives the polars-row encoding but not our
header plumbing, these will find it.
"""

import polars as pl
import pytest
from hypothesis import given, settings
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

from pl_row_encode import (
    decode,
    decode_peek,
    decode_series,
    encode,
    encode_series,
    get_header,
)

# Dtypes that polars-row encoding round-trips losslessly and that exercise the
# header/value plumbing: every int width, floats, bool, string, binary, temporal.
#
# The 8/16-bit ints only work because Cargo.toml enables the dtype-i8/i16/u8/u16 features
# on polars-core; without them the plugin can't reconstruct those columns across the FFI
# (regression-guarded by test_small_ints_roundtrip below).
ROUNDTRIP_DTYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.Boolean,
    pl.String,
    pl.Binary,
    pl.Date,
    pl.Datetime("us"),
    pl.Datetime("ns", "UTC"),
    pl.Time,
    pl.Duration("us"),
]

# A frame of 1..=5 columns, each a random allowed dtype, with nulls mixed in and a
# varying (non-empty) row count.
#
# min_size=1: a zero-row frame can't even reach our code -- pyo3-polars panics
# deserializing an empty input series across the plugin FFI boundary ("cannot create
# series from <dtype>"). That's an upstream limitation, not an encode/decode bug, so we
# don't exercise it here.
frames = dataframes(
    min_cols=1,
    max_cols=5,
    allowed_dtypes=ROUNDTRIP_DTYPES,
    min_size=1,
    max_size=8,
    allow_null=True,
)


@given(df=frames)
@settings(max_examples=300, deadline=None)
def test_decode_series_roundtrip(df: pl.DataFrame) -> None:
    tokens = encode_series(*[df[c] for c in df.columns])
    decoded = decode_series(tokens).struct.unnest()
    assert_frame_equal(decoded, df)


@given(df=frames)
@settings(max_examples=300, deadline=None)
def test_decode_explicit_header_roundtrip(df: pl.DataFrame) -> None:
    tokens = df.select(tok=encode(*df.columns))["tok"]
    header = get_header(tokens)
    out = (
        pl.DataFrame({"tok": tokens})
        .select(decode("tok", schema_header=header).alias("r"))
        .unnest("r")
    )
    assert_frame_equal(out, df)


@given(df=frames)
@settings(max_examples=200, deadline=None)
def test_decode_peek_roundtrip(df: pl.DataFrame) -> None:
    frame = pl.DataFrame({"tok": df.select(tok=encode(*df.columns))["tok"]})
    out = (
        frame.lazy().select(decode_peek(frame, "tok").alias("r")).unnest("r").collect()
    )
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, df)


@given(df=frames)
@settings(max_examples=200, deadline=None)
def test_header_is_stable_across_rows(df: pl.DataFrame) -> None:
    # The embedded header is identical in every token regardless of row contents.
    tokens = df.select(tok=encode(*df.columns))["tok"]
    headers = {bytes(get_header(t)) for t in tokens if t is not None}
    assert len(headers) <= 1


# --- edge cases / documented limitations (surfaced by the property tests above) -------


def test_empty_frame_encodes_and_decodes() -> None:
    # A zero-row frame encodes to an empty Binary column. The header-sniffing decoders
    # have no token to read, so they raise cleanly; an explicit header still round-trips.
    df = pl.DataFrame(
        {"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.String)}
    )
    src = pl.DataFrame({"a": [1], "b": ["x"]})  # supplies a header to decode against

    tokens = df.select(tok=encode("a", "b"))["tok"]
    assert tokens.dtype == pl.Binary
    assert tokens.len() == 0

    with pytest.raises(ValueError, match="all-null / empty"):
        decode_series(tokens)

    header = get_header(src.select(tok=encode("a", "b"))["tok"])
    out = (
        pl.DataFrame({"tok": tokens})
        .select(decode("tok", schema_header=header).alias("r"))
        .unnest("r")
    )
    assert out.height == 0
    assert out.schema == df.schema


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16])
def test_small_ints_roundtrip(dtype: pl.DataType) -> None:
    # 8/16-bit ints are opt-in cargo features (dtype-i8/i16/u8/u16) on polars-core;
    # without them the plugin panics reconstructing the column across the FFI with
    # "cannot create series from <dtype>". This guards that Cargo.toml keeps them enabled.
    df = pl.DataFrame({"a": pl.Series([1, None, 3], dtype=dtype)})
    out = decode_series(df.select(tok=encode("a"))["tok"]).struct.unnest()
    assert out["a"].dtype == dtype
    assert_frame_equal(out, df)


# Dtypes whose polars-core support is feature-gated (dtype-i128/decimal/array/categorical
# enable Enum). Each aborts the *process* with a non-unwinding panic when the feature is
# off, so these guard that Cargo.toml keeps them on. Construct one representative column
# each; nested/parametric fuzzing of these is left to the targeted cases below.
GATED_DTYPE_COLUMNS = {
    "Int128": pl.Series("a", [1, None, 3], dtype=pl.Int128),
    "Decimal": pl.Series("a", ["1.50", None, "3.25"], dtype=pl.Decimal(10, 2)),
    "Enum": pl.Series("a", ["x", None, "y"], dtype=pl.Enum(["x", "y"])),
    "Array": pl.Series("a", [[1, 2], None, [3, 4]], dtype=pl.Array(pl.Int64, 2)),
}


@pytest.mark.parametrize(
    "series", GATED_DTYPE_COLUMNS.values(), ids=GATED_DTYPE_COLUMNS.keys()
)
def test_gated_dtype_roundtrip(series: pl.Series) -> None:
    df = pl.DataFrame({"a": series})
    out = decode_series(df.select(tok=encode("a"))["tok"]).struct.unnest()
    assert out["a"].dtype == series.dtype
    assert_frame_equal(out, df)


def test_categorical_rejected_at_encode() -> None:
    # Categorical can't round-trip: the token carries only the physical integer category
    # keys, while the key->string mapping lives in a separate string cache that isn't
    # embedded. (Enum works because its categories live in the dtype, and therefore travel
    # inside our serialized schema header.) Rather than emit a token that panics on decode,
    # encode rejects Categorical up front with an actionable error.
    df = pl.DataFrame({"a": pl.Series(["x", "y"], dtype=pl.Categorical)})
    with pytest.raises(
        pl.exceptions.ComputeError, match="Categorical columns cannot be encoded"
    ):
        df.select(tok=encode("a"))


def test_nested_categorical_rejected_at_encode() -> None:
    # The rejection recurses through containers, so a Categorical buried in a List is
    # caught too (otherwise it would slip through and panic on decode).
    df = pl.DataFrame({"a": pl.Series([["x", "y"]], dtype=pl.List(pl.Categorical))})
    with pytest.raises(
        pl.exceptions.ComputeError, match="Categorical columns cannot be encoded"
    ):
        df.select(tok=encode("a"))
