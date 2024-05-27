import polars as pl
import polari as polari
import os
from functools import wraps
from time import time

def timing(runs=1):
    def decorator(f):
        @wraps(f)
        def wrap(*args, **kw):
            total_time = 0.0
            result = None
            for _ in range(runs):
                ts = time()
                result = f(*args, **kw)
                te = time()
                total_time += te - ts
            avg_time = total_time / runs
            print(
                "func:%r args:[%r, %r] avg_time: %2.4f sec over %d runs"
                % (f.__name__, args, kw, avg_time, runs)
            )
            return result

        return wrap

    return decorator

def process_raw(raw_dir: str, processed_dir: str, characters: int) -> None:
    files = os.listdir(raw_dir)

    languages = {
        "ru": "russian",
        "tu": "turkish",
        "fr": "french",
        "es": "spanish",
        "de": "german",
    }
    if characters > 0:
        text_expr = pl.col("column_1").str.slice(0, characters)
    else:
        text_expr = pl.col("column_1").cast(pl.Utf8)

    for file in files:
        lang = file.split("_")[0]
        print(lang)
        read_path = os.path.join(raw_dir, file)
        english_lang_name = languages[lang].capitalize()
        lf = pl.scan_csv(
            read_path,
            has_header=False,
            separator="|",
            ignore_errors=True,
            truncate_ragged_lines=True,
        ).select(
            text_expr.alias("text"),
            pl.lit(english_lang_name).alias("lang"),
        )
        output_path = os.path.join(
            processed_dir, os.path.splitext(file)[0] + ".parquet"
        )
        lf.sink_parquet(output_path)


def scan_parquets(dir: str, n_rows_per_file: str) -> pl.LazyFrame:
    files = os.listdir(dir)
    files = [os.path.join(dir, file) for file in files]
    lfs = []
    for file in files:
        lf = pl.scan_parquet(file, n_rows=n_rows_per_file)
        lfs.append(lf)
    lf = pl.concat(lfs)
    return lf


# @timing(runs=1)
def misleading_benchmark(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = (
        (
            lf.select(
                "lang",
                what_lang_eng_name=polari.what_lang("text", lang_format="eng_name"),
            )
        )
        .with_columns(
            accurate=pl.col("lang").eq(pl.col("what_lang_eng_name")).fill_null(False)
        )
        .group_by("lang", "accurate")
        .agg(pl.len().alias("lang_accurate_count"))
        .with_columns(pl.sum("lang_accurate_count").over("lang").alias("lang_count"))
        .filter(pl.col("accurate"))
        .select(
            "lang",
            "lang_count",
            accurate=pl.col("lang_accurate_count").truediv(pl.col("lang_count")),
        )
        .sort("lang")
    )
    return lf.collect()


if __name__ == "__main__":
    data_folder = "data"
    doc_type = "summaries"
    raw_dir = os.path.join(data_folder, "raw", doc_type)
    processed_dir = os.path.join(data_folder, "processed", doc_type)
    # process_raw(raw_dir=raw_dir, processed_dir=processed_dir, characters=-1)
    lf = scan_parquets(processed_dir, 5)
    # lf = pl.scan_parquet(os.path.join(processed_dir,"*"))
    # lf = lf.pipe(misleading_benchmark)
    include_langs = ["russian","turkish","french", "spanish", "german"]
    lf = lf.select(
        "text",
        polari.detect_lang("text", algorithm="which_lang").alias("which_lang"),
        # polari.detect_lang("text", algorithm="what_lang", include_langs=include_langs).alias("what_lang"),
        # polari.detect_confidence("text", algorithm="what_lang").alias("what_lang_confidence"),
        # polari.detect_lang("text", algorithm="lingua", in_parallel=True, include_langs=include_langs).alias("lingua_lang_in_parallel"),
        # polari.detect_lang("text", algorithm="lingua", in_parallel=True, include_langs=include_langs, low_accuracy=True).alias("lingua_lang_in_parallel_low_accuracy"),
        # polari.detect_lang("text", algorithm="lingua", include_langs=include_langs).alias("lingua_lang"),
        # polari.detect_script("text").alias("script"),
        # polari.detect_lang_confidence("text",algorithm="what_lang", include_langs=include_langs).alias("what_lang_confidence"),
        # polari.detect_lang_confidence("text", algorithm="lingua", include_langs=include_langs).alias("lingua_lang_confidence"),
        polari.sentiment("text", output_type="compound").alias("lingua_lang_confidence"),

    ).collect()
    print(lf)
