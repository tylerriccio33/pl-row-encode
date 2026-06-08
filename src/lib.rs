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

/// Categorical's category->string mapping lives in an external string cache, not in the
/// dtype or the row bytes, so a token can't carry it -- decoding would panic in
/// `cat_to_str`. Reject it (recursing through nested containers) at encode time with an
/// actionable message instead of emitting a token that only blows up on decode. Enum is
/// fine: its categories live in the dtype and ride along in the serialized header.
fn reject_categorical(dtype: &DataType) -> PolarsResult<()> {
    match dtype {
        DataType::Categorical(_, _) => polars_bail!(
            ComputeError:
            "Categorical columns cannot be encoded: the category mapping is not embeddable \
             in a token. Use Enum, or cast to String before encoding."
        ),
        DataType::List(inner) => reject_categorical(inner),
        DataType::Array(inner, _) => reject_categorical(inner),
        DataType::Struct(fields) => {
            for f in fields {
                reject_categorical(f.dtype())?;
            }
            Ok(())
        },
        _ => Ok(()),
    }
}

#[polars_expr(output_type=Binary)]
fn row_encode(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        polars_bail!(ComputeError: "row_encode requires at least one input column");
    }

    let columns: Vec<Column> = inputs.iter().cloned().map(Column::from).collect();
    let fields: Vec<Field> = inputs.iter().map(|s| s.field().into_owned()).collect();

    for field in &fields {
        reject_categorical(field.dtype())?;
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
