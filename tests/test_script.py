from polari import detect_script
import polars as pl
from polars.testing import assert_frame_equal


def test_detect_script():
    df = pl.DataFrame(
        [
            (
                "Был осуждён по делу петрашевцев на четыре года каторги, отбывал своё наказание в военном городе Омске.",
                'Attunement is "our way of finding ourselves thrust into the world". It can also be translated as "disposition" or "affectedness".',
                "世界規模で展開し、世界で初めてフランチャイズビジネスを創始した。現在はペプシコ社から分派したヤム・ブランズの傘下である。",
                None,
            ),
        ],
        schema={"text": pl.Utf8},
    )

    test_eager = df.select(
        language=detect_script(
            "text",
        )
    )
    test_lazy = (
        df.lazy()
        .select(
            language=detect_script(
                "text",
            ),
        )
        .collect()
    )

    exp = pl.DataFrame(
        [
            ("Cyrillic", "Latin", "Katakana", None),
        ],
        schema={"language": pl.Utf8},
    )

    assert_frame_equal(test_eager, exp)
    assert_frame_equal(test_lazy, exp)
