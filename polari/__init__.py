from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

import polars as pl

from polari.utils import parse_into_expr, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

## TODO : move different behaviours into own name spaces

# def what_lang_reliable(expr: IntoExpr) -> pl.Expr:
#     expr = parse_into_expr(expr)
#     return expr.register_plugin(
#         lib=lib,
#         symbol="what_lang_reliable",
#         is_elementwise=True,
#     )


def capitalize_langs(langs: List[str]) -> List[str]:
    return [lang.capitalize() for lang in langs]


def detect_lang(
    expr: IntoExpr,
    *,
    algorithm: str,
    include_langs: List = [],
    exclude_langs: List = [],
    in_parallel: bool = False,
    low_accuracy: bool = False,
) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="detect_language",
        is_elementwise=True,
        kwargs={
            "algorithm": algorithm,
            "include_langs": capitalize_langs(include_langs),
            "exclude_langs": capitalize_langs(exclude_langs),
            "in_parallel": in_parallel,
            "low_accuracy": low_accuracy,
        },
    )


def detect_lang_confidence(
    expr: IntoExpr,
    *,
    algorithm: str,
    include_langs: List = [],
    exclude_langs: List = [],
    low_accuracy: bool = False,
) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="detect_language_confidence",
        is_elementwise=True,
        kwargs={
            "algorithm": algorithm,
            "include_langs": capitalize_langs(include_langs),
            "exclude_langs": capitalize_langs(exclude_langs),
            "in_parallel": False,
            "low_accuracy": low_accuracy,
        },
    )


def detect_script(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="detect_script",
        is_elementwise=True,
    )


def get_sentiment(expr: IntoExpr, output_type: str) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="get_sentiment",
        is_elementwise=True,
        kwargs={"score_type": output_type},
    )
