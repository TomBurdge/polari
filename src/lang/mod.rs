#![allow(clippy::unused_unit)]
use lingua::LanguageDetectorBuilder;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use whatlang::{Detector, Lang};
use whichlang::detect_language;
pub mod builder;
use crate::lang::builder::{process_language_list, process_language_list_lingua};

#[derive(Deserialize)]
struct AddAlgoKwargs {
    algorithm: String,
    include_langs: Vec<String>,
    exclude_langs: Vec<String>,
    in_parallel: bool,
    low_accuracy: bool,
}

// TODO: stricter builders.
enum DetectorType {
    WhatLang(whatlang::Detector),
    LinguaInParallel(LinguaInParallel),
    LinguaSequential(LinguaSequential),
    WhichLang,
}
struct LinguaInParallel {
    detector: lingua::LanguageDetector,
}

struct LinguaSequential {
    detector: lingua::LanguageDetector,
}

enum Algorithm {
    WhatLang {
        include_langs: Vec<String>,
        exclude_langs: Vec<String>,
    },
    Lingua {
        low_accuracy: bool,
        include_langs: Vec<String>,
        exclude_langs: Vec<String>,
        in_parallel: bool,
    },
    WhichLang,
}

impl Algorithm {
    fn from_kwargs(kwargs: &AddAlgoKwargs) -> Result<Self, String> {
        // TODO: match multiple in the kwargs, then return an error if is wrong combo
        // e.g. lingua with confidence
        match kwargs.algorithm.as_str() {
            "what_lang" => Ok(Algorithm::WhatLang{
                include_langs: kwargs.include_langs.clone(),
                exclude_langs: kwargs.exclude_langs.clone()
            }),
            "lingua" => Ok(Algorithm::Lingua {
                low_accuracy: kwargs.low_accuracy.clone(),
                include_langs: kwargs.include_langs.clone(),
                exclude_langs: kwargs.exclude_langs.clone(),
                in_parallel: kwargs.in_parallel.clone(),
            }),
            "which_lang" => Ok(Algorithm::WhichLang),
            _ => Err(format!(
                "Invalid algorithm: {}. Must be one of: what_lang, lingua, lingua_high_accuracy, lingua_low_accuracy.",
                kwargs.algorithm
            )),
        }
    }

    fn build(&self) -> Result<DetectorType, PolarsError> {
        match self {
            Algorithm::WhatLang {
                include_langs,
                exclude_langs,
            } => {
                let allowlist = if include_langs.is_empty() {
                    vec![]
                } else {
                    process_language_list(include_langs)?
                };
                let denylist = if exclude_langs.is_empty() {
                    vec![]
                } else {
                    process_language_list(exclude_langs)?
                };

                let detector = if !allowlist.is_empty() {
                    Detector::with_allowlist(allowlist)
                } else if !denylist.is_empty() {
                    Detector::with_denylist(denylist)
                } else {
                    Detector::new()
                };

                Ok(DetectorType::WhatLang(detector))
            }
            Algorithm::Lingua {
                low_accuracy,
                include_langs,
                exclude_langs,
                in_parallel,
            } => {
                // panics if less than two languages supplied, so should handle that
                let allowlist = if include_langs.is_empty() {
                    vec![]
                } else {
                    process_language_list_lingua(include_langs)?
                };
                let denylist = if exclude_langs.is_empty() {
                    vec![]
                } else {
                    process_language_list_lingua(exclude_langs)?
                };

                let mut detector = if !allowlist.is_empty() {
                    LanguageDetectorBuilder::from_languages(&allowlist)
                } else if !denylist.is_empty() {
                    LanguageDetectorBuilder::from_all_languages_without(&denylist)
                } else {
                    LanguageDetectorBuilder::from_all_languages()
                };

                if *low_accuracy {
                    detector.with_low_accuracy_mode();
                }

                if *in_parallel {
                    Ok(DetectorType::LinguaInParallel(LinguaInParallel {
                        detector: detector.build(),
                    }))
                } else {
                    Ok(DetectorType::LinguaSequential(LinguaSequential {
                        detector: detector.build(),
                    }))
                }
            }
            Algorithm::WhichLang => Ok(DetectorType::WhichLang),
        }
    }
}

trait DetectorTrait {
    fn detect_language(&self, inputs: &[Series]) -> PolarsResult<Series>;
}
trait ConfidenceTrait {
    fn detect_language_confidence(&self, inputs: &[Series]) -> PolarsResult<Series>;
}

impl DetectorTrait for whatlang::Detector {
    fn detect_language(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let ca: &StringChunked = inputs[0].str()?;
        let mut result_builder = StringChunkedBuilder::new("lang", ca.len());
        ca.into_iter().for_each(|op_s| {
            if let Some(s) = op_s {
                if let Some(info) = self.detect_lang(s) {
                    result_builder.append_value(info.eng_name());
                } else {
                    result_builder.append_null();
                }
            } else {
                result_builder.append_null();
            }
        });
        let out = result_builder.finish();
        Ok(out.into_series())
    }
}
impl ConfidenceTrait for whatlang::Detector {
    fn detect_language_confidence(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let ca: &StringChunked = inputs[0].str()?;
        let out = ca
            .into_iter()
            .map(|opt_value| {
                opt_value.map(|value| {
                    self.detect(value)
                        .map(|d| d.confidence())
                        .unwrap_or_else(|| 0.0)
                })
            })
            .collect::<ChunkedArray<Float64Type>>();
        Ok(out.into_series())
    }
}

impl DetectorTrait for LinguaSequential {
    fn detect_language(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let ca: &StringChunked = inputs[0].str()?;
        let mut lingua_builder = StringChunkedBuilder::new("lang", ca.len());
        ca.into_iter().for_each(|op_s| {
            if let Some(s) = op_s {
                if let Some(text) = self.detector.detect_language_of(s) {
                    lingua_builder.append_value(text.to_string());
                } else {
                    lingua_builder.append_null();
                }
            } else {
                lingua_builder.append_null();
            }
        });
        let out = lingua_builder.finish();
        Ok(out.into_series())
    }
}

impl ConfidenceTrait for LinguaSequential {
    fn detect_language_confidence(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let ca: &StringChunked = inputs[0].str()?;
        let confidences = ca
            .into_iter()
            .map(|opt_value| {
                opt_value.and_then(|value| {
                    self.detector
                        .detect_language_of(value)
                        .map(|lang| self.detector.compute_language_confidence(value, lang))
                })
            })
            .collect::<ChunkedArray<Float64Type>>();
        Ok(confidences.into_series())
    }
}

impl DetectorTrait for LinguaInParallel {
    fn detect_language(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let ca: &StringChunked = inputs[0].str()?;
        let vec: Vec<String> = ca
            .into_iter()
            .map(|opt_str| opt_str.unwrap_or_default().to_string())
            .collect();
        let langs = self.detector.detect_languages_in_parallel_of(&vec);
        let out = langs
            .into_iter()
            .map(|opt_lang| opt_lang.map(|lang| lang.to_string()))
            .collect::<ChunkedArray<StringType>>();
        Ok(out.into_series())
    }
}

fn which_lang_detect_language(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let mut which_lang_builder = StringChunkedBuilder::new("lang", ca.len());
    ca.into_iter().for_each(|op_s| {
        if let Some(s) = op_s {
            let lang = detect_language(s);
            if let Some(lang) = Lang::from_code(lang.three_letter_code()) {
                which_lang_builder.append_value(lang.eng_name());
            } else {
                which_lang_builder.append_null();
            }
        } else {
            which_lang_builder.append_null();
        }
    });
    let out = which_lang_builder.finish();
    Ok(out.into_series())
}

impl DetectorTrait for DetectorType {
    fn detect_language(&self, inputs: &[Series]) -> PolarsResult<Series> {
        match self {
            DetectorType::WhatLang(detector) => detector.detect_language(inputs),
            DetectorType::LinguaInParallel(detector) => detector.detect_language(inputs),
            DetectorType::LinguaSequential(detector) => detector.detect_language(inputs),
            DetectorType::WhichLang => which_lang_detect_language(inputs),
        }
    }
}

#[polars_expr(output_type = String)]
fn detect_language(inputs: &[Series], kwargs: AddAlgoKwargs) -> PolarsResult<Series> {
    let algorithm =
        Algorithm::from_kwargs(&kwargs).map_err(|e| PolarsError::ComputeError(e.into()))?;
    let detector = algorithm.build()?;
    detector.detect_language(inputs)
}

impl ConfidenceTrait for DetectorType {
    fn detect_language_confidence(&self, inputs: &[Series]) -> PolarsResult<Series> {
        match self {
            DetectorType::WhatLang(detector) => detector.detect_language_confidence(inputs),
            DetectorType::LinguaSequential(detector) => detector.detect_language_confidence(inputs),
            DetectorType::LinguaInParallel(_detector) => {
                let err_message = "The algorithm does not implement a confidence value".to_string();
                Err(PolarsError::ComputeError(err_message.into()))
            }
            DetectorType::WhichLang => {
                let err_message = "The algorithm does not implement a confidence value".to_string();
                Err(PolarsError::ComputeError(err_message.into()))
            }
        }
    }
}

#[polars_expr(output_type = Float64)]
fn detect_language_confidence(inputs: &[Series], kwargs: AddAlgoKwargs) -> PolarsResult<Series> {
    let algorithm =
        Algorithm::from_kwargs(&kwargs).map_err(|e| PolarsError::ComputeError(e.into()))?;
    let detector = algorithm.build()?;
    detector.detect_language_confidence(inputs)
}
