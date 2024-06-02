import polars as pl
from typing import Tuple, Union, Any

extra_dependencies = """duckdb==0.10.3
polars==0.20.30
pyarrow==16.1.0"""
try:
    import duckdb # type: ignore
except Exception:
    ImportError(
        "Could not find the duckdb module.",
        "you need to install the example dependencies to load the example dataset.",
        "The extra dependencies are:",
        extra_dependencies,
    )


def load_aya(
    limit: Union[int, Any] = None, language_subset: Union[Tuple[str], Any] = None
) -> pl.LazyFrame:
    try:
        dataset = duckdb.sql(
            """SELECT
                    inputs
                    , language 
                FROM 'hf://datasets/CohereForAI/aya_dataset/data/train-00000-of-00001.parquet'
            """
        )
        if language_subset:
            dataset = dataset.filter(f"language in {language_subset}")
        if limit:
            dataset = dataset.limit(limit)
        lf = dataset.pl().lazy()
    except Exception as e:
        raise ImportError(
            "Could not load the data.",
            "You like likely do not have the extra dependencies, which do not come built-in with polari.",
            "The extra dependencies are:",
            extra_dependencies,
            "here is the error message:",
            e
        )
    return lf
