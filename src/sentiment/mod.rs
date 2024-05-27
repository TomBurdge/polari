#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::HashMap;
use std::str::FromStr;
use vader_sentiment::SentimentIntensityAnalyzer;

// TODO add checking on output type through an enum
// builder which checks the kwarg input first - stricter.

#[derive(Deserialize)]
struct AddOutputTypeKwarg {
    score_type: String,
}

#[derive(Debug)]
enum ScoreType {
    Compound,
    Positive,
    Neutral,
    Negative,
}
impl FromStr for ScoreType {
    type Err = PolarsError;

    fn from_str(s: &str) -> Result<ScoreType, Self::Err> {
        match s {
            "compound" => Ok(ScoreType::Compound),
            "pos" | "positive" => Ok(ScoreType::Positive),
            "neu" | "neutral" => Ok(ScoreType::Neutral),
            "neg" | "negative" => Ok(ScoreType::Negative),
            _ => Err(PolarsError::ComputeError(s.to_string().into())),
        }
    }
}
impl ScoreType {
    fn as_str(&self) -> &'static str {
        match self {
            ScoreType::Compound => "compound",
            ScoreType::Positive => "pos",
            ScoreType::Neutral => "neu",
            ScoreType::Negative => "neg",
        }
    }
}

#[polars_expr(output_type=String)]
fn get_sentiment(inputs: &[Series], kwargs: AddOutputTypeKwarg) -> PolarsResult<Series> {
    let score_type_arg = &kwargs.score_type;
    let score_type = ScoreType::from_str(&score_type_arg)?.as_str();
    let ca: &StringChunked = inputs[0].str()?;
    let analyzer = SentimentIntensityAnalyzer::new();
    let scores: Vec<Option<f64>> = ca
        .into_iter()
        .map(|opt_s| {
            opt_s.and_then(|s| {
                let scores: HashMap<&str, f64> = analyzer.polarity_scores(s);
                scores.get(&score_type).cloned()
            })
        })
        .collect();
    let out = Float64Chunked::from_slice_options("scores", &scores);
    Ok(out.into_series())
}
