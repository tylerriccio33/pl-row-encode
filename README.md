# pl-row-encode

A Polars plugin for **row-level, type-preserving encode/decode**.

`encode(*cols)` packs a set of columns into a single `Binary` column where each value is an
opaque, **self-describing** token: the [`polars-row`](https://docs.rs/polars-row) encoding of
the row, prefixed with an embedded schema header. `decode_series(...)` reverses it back into a
`Struct` — recovering the original dtypes — **without needing any external schema**.

```
DataFrame
  -> encode(*cols)        # Arrow columns -> canonical typed row bytes (Binary)
  -> vendor (opaque bytes)
  -> decode_series(...)   # row bytes -> Struct -> original typed columns
  -> DataFrame
```

The vendor only ever holds opaque bytes; the type information rides along inside each token.

## Token layout

Each `Binary` value is:

```
[ u32 header_len (LE) ][ header bytes ][ row bytes ]
```

`header` is a bincode-serialized `Vec<Field>` (logical schema); `row bytes` is the
unordered `polars-row` encoding of that single row. Embedding the header per value makes
every token independently decodable.

## Usage

```python
import polars as pl
from pl_row_encode import encode, decode_series

df = pl.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})

tokens = df.select(tok=encode("id", "name"))["tok"]   # dtype: Binary
# ... hand `tokens` to a vendor, get them back ...

decoded = decode_series(tokens).struct.unnest()        # back to id / name with dtypes
```

For the lazy engine, the output `Struct` dtype must be known up front, so pass a token's
header explicitly:

```python
from pl_row_encode import decode
header = ...  # the [u32 len][header] prefix of any token
lf.select(decode("tok", schema_header=header)).collect()
```

## Development

```bash
make develop   # build the Rust extension into the venv (uv run maturin develop)
make test      # build + run pytest
make lint      # ruff + ty
```

The first `make develop` compiles the full Polars Rust workspace and takes a few minutes;
subsequent builds are incremental and fast.

## Notes / limitations

- Built on `polars-row`, the same machinery Polars uses internally for sort/group-by row
  encoding — lossless for primitive, string, boolean, temporal, and nested types.
- `decode_series` infers the schema from the first non-null token, so an all-null/empty
  Series needs the explicit `decode(schema_header=...)` form.
- The maturin build prints a harmless `Couldn't find the symbol PyInit__internal` warning;
  this is expected for Polars expression plugins (they are dlopened by Polars, not imported).
