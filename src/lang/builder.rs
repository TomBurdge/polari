use isolang::Language as IsoLanguage;
use lingua::Language;
use polars::prelude::*;
use whatlang::Lang;

pub fn process_language_list_lingua(langs: &[String]) -> Result<Vec<Language>, PolarsError> {
    langs
        .iter()
        .map(|lang| {
            lang.parse().map_err(|_| {
                PolarsError::ComputeError(format!("Language not found: {}", lang).into())
            })
        })
        .collect()
}

fn get_language_code(language_name: &str) -> Result<Lang, PolarsError> {
    let iso_lang = IsoLanguage::from_name(language_name).ok_or_else(|| {
        PolarsError::ComputeError(format!("Language not found: {}", language_name).into())
    })?;
    let code = iso_lang.to_639_3();
    Lang::from_code(code).ok_or_else(|| {
        PolarsError::ComputeError(format!("Language not found: {}", language_name).into())
    })
}

pub fn process_language_list(lang_list: &[String]) -> Result<Vec<Lang>, PolarsError> {
    lang_list.iter().map(|s| get_language_code(s)).collect()
}
