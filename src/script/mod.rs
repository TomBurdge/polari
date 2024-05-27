#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use whatlang::detect_script as what_lang_detect_script;

#[polars_expr(output_type=String)]
fn detect_script(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let mut what_script_builder = StringChunkedBuilder::new("lang", ca.len());
    ca.into_iter().for_each(|op_s| {
        if let Some(s) = op_s {
            if let Some(info) = what_lang_detect_script(s) {
                what_script_builder.append_value(info.to_string());
            } else {
                what_script_builder.append_null();
            }
        } else {
            what_script_builder.append_null();
        }
    });
    let out = what_script_builder.finish();
    Ok(out.into_series())
}
