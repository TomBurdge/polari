import polars as pl
from typing import Tuple
import duckdb


def load_aya(limit: int, language_subset: Tuple[str] = None) -> pl.LazyFrame:
    dataset = duckdb.sql(
        """SELECT
                inputs
                , language 
            FROM 'hf://datasets/CohereForAI/aya_dataset/data/train-00000-of-00001.parquet'
        """
    )
    if language_subset:
        dataset = dataset.filter(f"language in {language_subset}")
    dataset = dataset.limit(limit)
    lf = dataset.pl().lazy()
    return lf
