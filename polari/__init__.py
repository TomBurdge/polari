from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

import polars as pl
from polars.plugins import register_plugin_function

from polari.utils import parse_into_expr, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

# more plugin ideas: units converter. runtime_units (or other)
# polars faker: creates fake data with rust faker crate


def capitalize_langs(langs: List[str]) -> List[str]:
    return [lang.capitalize() for lang in langs]


def detect_lang(
    expr: IntoExpr,
    *,
    algorithm: str = "which_lang",
    include_langs: List = [],
    exclude_langs: List = [],
    in_parallel: bool = False,
    low_accuracy: bool = False,
) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=lib,  # type: ignore
        function_name="detect_language",
        args=expr,
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
    return register_plugin_function(
        plugin_path=lib,  # type: ignore
        function_name="detect_language_confidence",
        args=expr,
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
    return register_plugin_function(
        plugin_path=lib,  # type: ignore
        function_name="detect_script",
        args=expr,
        is_elementwise=True,
    )


def get_sentiment(expr: IntoExpr, output_type: str) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=lib,  # type: ignore
        function_name="get_sentiment",
        args=expr,
        is_elementwise=True,
        kwargs={"score_type": output_type},
    )
