mod lang;
mod script;
mod sentiment;
mod utils;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

use pyo3::types::PyModule;
use pyo3::{pymodule, PyResult, Python};

// TODO:
// add name -> native name/code name with a crate.
// re-add return code option
// do I want to spend the time on this...

// TODO: add is_reliable, with confidence threshold
// for what_lang, was quite simple - number for the threshold passed in as arg.

// TODO: create stemmer module for multiple langs

// TODO: create tokenizer module (from polars ds issue)

#[pymodule]
fn polari(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
