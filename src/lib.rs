use polars::prelude::*;
use polars_arrow::array::{BinaryArray, MutableBinaryArray};
use polars_core::chunked_array::ops::row_encode::{encode_rows_unordered, row_encoding_decode};
use polars_row::RowEncodingOptions;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Token layout for each encoded row:
///   [ u32 header_len (LE) ][ header bytes ][ row bytes ]
/// where `header` is a bincode-serialized `Vec<Field>` (the logical schema) and
/// `row bytes` is the polars-row unordered encoding of that single row. Embedding the
/// header per value makes every token independently self-describing.
fn serialize_header(fields: &[Field]) -> PolarsResult<Vec<u8>> {
    bincode::serialize(&fields.to_vec())
        .map_err(|e| polars_err!(ComputeError: "failed to serialize schema header: {e}"))
}

fn deserialize_header(bytes: &[u8]) -> PolarsResult<Vec<Field>> {
    bincode::deserialize(bytes)
        .map_err(|e| polars_err!(ComputeError: "failed to deserialize schema header: {e}"))
}

#[derive(Deserialize)]
struct DecodeKwargs {
    /// The `[u32 len][header]` prefix lifted from a token, forwarded by the Python wrapper
    /// so the output Struct dtype can be resolved before any data is seen.
    #[serde(with = "serde_bytes")]
    schema_header: Vec<u8>,
}

/// Pull the `Vec<Field>` schema out of a `[u32 len][header][..]` prefixed buffer.
fn fields_from_prefixed(buf: &[u8]) -> PolarsResult<Vec<Field>> {
    if buf.len() < 4 {
        polars_bail!(ComputeError: "token too short to contain a schema header");
    }
    let hlen = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let end = 4 + hlen;
    if buf.len() < end {
        polars_bail!(ComputeError: "token header length exceeds buffer");
    }
    deserialize_header(&buf[4..end])
}

/// Classify a dtype against three tiers to uphold the invariant that anything we encode
/// can be decoded (see CONTRACT.md):
///
///   * **known-good** -- dtypes whose polars-row encode/decode we've verified round-trips
///     losslessly (the allowlist below, guarded by the dtype-matrix property tests). These
///     encode silently.
///   * **known-bad** -- `Categorical`. Its category->string mapping lives in an external
///     string cache, not in the dtype or row bytes, so a token can't carry it; decoding
///     panics in `cat_to_str`. Hard-rejected with an actionable message instead of
///     emitting a token that blows up on decode. (Enum is fine: its categories live in the
///     dtype and ride along in the serialized header.)
///   * **unknown** -- anything else (new/exotic dtypes we haven't vetted). We can't safely
///     probe it (the probe is the dangerous decode), so its name is collected and the
///     caller emits a warning: the token may fail or panic on decode.
///
/// Recurses through `List` / `Array` / `Struct` so a nested offender is caught too. The
/// allowlist is keyed to the compiled polars crate version (Cargo.toml), not the user's
/// runtime polars, since the plugin decodes with its own embedded `polars-row`.
fn classify_dtype(dtype: &DataType, unknown: &mut Vec<String>) -> PolarsResult<()> {
    match dtype {
        DataType::Categorical(_, _) => polars_bail!(
            ComputeError:
            "Categorical columns cannot be encoded: the category mapping is not embeddable \
             in a token. Use Enum, or cast to String before encoding."
        ),

        // Allowlist: leaves verified to round-trip (see tests/test_property.py).
        DataType::Boolean
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal(_, _)
        | DataType::String
        | DataType::Binary
        | DataType::Date
        | DataType::Time
        | DataType::Datetime(_, _)
        | DataType::Duration(_)
        | DataType::Enum(_, _) => Ok(()),

        // Containers: known-good iff their inner dtype(s) are.
        DataType::List(inner) | DataType::Array(inner, _) => classify_dtype(inner, unknown),
        DataType::Struct(fields) => {
            for f in fields {
                classify_dtype(f.dtype(), unknown)?;
            }
            Ok(())
        }

        other => {
            unknown.push(format!("{other}"));
            Ok(())
        }
    }
}

/// Emit a Python `UserWarning` naming dtypes that aren't on the known-good allowlist.
/// Reaching Python from inside a polars expr requires grabbing the GIL; failures to warn
/// are swallowed (a warning must never break an encode).
fn warn_unknown_dtypes(names: &[String]) {
    use pyo3::prelude::*;

    let msg = format!(
        "pl_row_encode: dtype(s) [{}] are not in the known round-trip-safe set; the \
         resulting token may fail or panic on decode. If decode works, please report it \
         so the dtype can be allowlisted.",
        names.join(", "),
    );
    let _ = Python::attach(|py| -> PyResult<()> {
        py.import("warnings")?.call_method1("warn", (msg,))?;
        Ok(())
    });
}

#[polars_expr(output_type=Binary)]
fn row_encode(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        polars_bail!(ComputeError: "row_encode requires at least one input column");
    }

    let columns: Vec<Column> = inputs.iter().cloned().map(Column::from).collect();
    let fields: Vec<Field> = inputs.iter().map(|s| s.field().into_owned()).collect();

    let mut unknown: Vec<String> = Vec::new();
    for field in &fields {
        classify_dtype(field.dtype(), &mut unknown)?;
    }
    if !unknown.is_empty() {
        warn_unknown_dtypes(&unknown);
    }

    let header = serialize_header(&fields)?;
    let hlen = (header.len() as u32).to_le_bytes();

    let encoded: BinaryOffsetChunked = encode_rows_unordered(&columns)?;

    let tokens: Vec<Vec<u8>> = encoded
        .into_iter()
        .map(|opt| {
            let row = opt.unwrap_or(&[]);
            let mut token = Vec::with_capacity(4 + header.len() + row.len());
            token.extend_from_slice(&hlen);
            token.extend_from_slice(&header);
            token.extend_from_slice(row);
            token
        })
        .collect();

    let out = BinaryChunked::from_iter_values(
        inputs[0].name().clone(),
        tokens.iter().map(|t| t.as_slice()),
    );
    Ok(out.into_series())
}

fn decode_output(_input: &[Field], kwargs: DecodeKwargs) -> PolarsResult<Field> {
    let fields = fields_from_prefixed(&kwargs.schema_header)?;
    Ok(Field::new("row_decode".into(), DataType::Struct(fields)))
}

#[polars_expr(output_type_func_with_kwargs=decode_output)]
fn row_decode(inputs: &[Series], kwargs: DecodeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let fields = fields_from_prefixed(&kwargs.schema_header)?;

    // Strip the schema header off each token, keeping just the row bytes, and rebuild a
    // BinaryOffset array for polars-row to decode.
    let mut rows = MutableBinaryArray::<i64>::with_capacity(ca.len());
    for opt in ca.into_iter() {
        let tok = opt.ok_or_else(|| polars_err!(ComputeError: "cannot decode a null token"))?;
        if tok.len() < 4 {
            polars_bail!(ComputeError: "token too short to contain a schema header");
        }
        let hlen = u32::from_le_bytes(tok[0..4].try_into().unwrap()) as usize;
        let start = 4 + hlen;
        if tok.len() < start {
            polars_bail!(ComputeError: "token header length exceeds buffer");
        }
        rows.push(Some(&tok[start..]));
    }

    let arr: BinaryArray<i64> = rows.into();
    let row_ca = BinaryOffsetChunked::with_chunk(inputs[0].name().clone(), arr);

    let opts = vec![RowEncodingOptions::new_unsorted(); fields.len()];
    let decoded = row_encoding_decode(&row_ca, &fields, &opts)?;
    Ok(decoded.into_series())
}
