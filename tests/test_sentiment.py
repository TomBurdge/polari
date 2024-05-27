from polari import get_sentiment
import polars as pl
from polars.testing import assert_frame_equal
import pytest
import operator
from typing import Callable


@pytest.mark.parametrize(
    """text, output_type, op, value""",
    [
        ("This was great! I love it.", "compound", operator.gt, 0),
        (
            "Awful. The worst thing ever. I hate it. Dreadful",
            "compound",
            operator.lt,
            0,
        ),
        ("This was great! I love it.", "positive", operator.gt, 0.5),
        (
            "Awful. The worst thing ever. I hate it. Dreadful",
            "positive",
            operator.lt,
            0.5,
        ),
        ("This was great! I love it.", "pos", operator.gt, 0.5),
        ("Awful. The worst thing ever. I hate it. Dreadful", "pos", operator.lt, 0.5),
        ("This was great! I love it.", "negative", operator.lt, 0.5),
        (
            "Awful. The worst thing ever. I hate it. Dreadful",
            "negative",
            operator.gt,
            0.5,
        ),
        ("This was great! I love it.", "neg", operator.lt, 0.5),
        ("Awful. The worst thing ever. I hate it. Dreadful", "neg", operator.gt, 0.5),
        ("This was great! I love it.", "neutral", operator.lt, 0.5),
        (
            "Awful. The worst thing ever. I hate it. Dreadful",
            "neutral",
            operator.lt,
            0.5,
        ),
        ("This was great! I love it.", "neu", operator.lt, 0.5),
        ("Awful. The worst thing ever. I hate it. Dreadful", "neu", operator.lt, 0.5),
    ],
)
def test_which_lang_compound(text: str, output_type: str, op: Callable, value: float):
    df = pl.DataFrame([text], schema={"text": pl.Utf8})

    test_eager = df.select(compound=get_sentiment("text", output_type)).select(
        positive=op(pl.col("compound"), value)
    )
    test_lazy = (
        df.lazy()
        .select(compound=get_sentiment("text", output_type))
        .select(positive=op(pl.col("compound"), value))
        .collect()
    )

    exp = pl.DataFrame([True], schema={"positive": pl.Boolean})
    assert_frame_equal(test_eager, exp)
    assert_frame_equal(test_lazy, exp)
