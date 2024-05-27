import pytest
from polari import detect_lang, detect_lang_confidence
import polars as pl
from polars.testing import assert_frame_equal
from collections import OrderedDict


# there is some kind of inconsistency to do with chinese here
# TODO: troubleshoot chinese/mandarin chinese via the langs.
# could be an inconsistency between the crates
# Mandarin Chinese needs to be upper case for both
@pytest.mark.parametrize(
    """algorithm, include_langs, exclude_langs, in_parallel, low_accuracy""",
    [
        ("which_lang", [], [], False, False),
        ("what_lang", [], [], False, False),
        ("what_lang", ["russian", "english", "japanese", "french"], [], False, False),
        ("what_lang", [], ["spanish"], False, False),
        ("lingua", [], [], False, False),
        ("lingua", [], [], True, False),
        ("lingua", [], [], False, True),
        ("lingua", [], [], True, True),
        ("lingua", ["russian", "english", "japanese", "french"], [], False, False),
        ("lingua", ["russian", "english", "japanese", "french"], [], True, False),
        ("lingua", ["russian", "english", "japanese", "french"], [], False, True),
        (
            "lingua",
            ["russian", "english", "japanese", "french"],
            ["spanish"],
            False,
            True,
        ),
        ("lingua", [], ["spanish"], False, True),
        ("lingua", ["russian", "english", "japanese", "french"], [], True, True),
    ],
)
def test_which_lang(
    algorithm: str,
    include_langs: list,
    exclude_langs: list,
    in_parallel: bool,
    low_accuracy: bool,
):
    df = pl.DataFrame(
        [
            (
                "Après une enfance difficile, il est élève d'une école d'officiers et se lie avec le mouvement progressiste de Saint-Pétersbourg.",
                "Был осуждён по делу петрашевцев на четыре года каторги, отбывал своё наказание в военном городе Омске.",
                'Attunement is "our way of finding ourselves thrust into the world". It can also be translated as "disposition" or "affectedness".',
                "世界規模で展開し、世界で初めてフランチャイズビジネスを創始した。現在はペプシコ社から分派したヤム・ブランズの傘下である。",
                None,
            ),
        ],
        schema={"text": pl.Utf8},
    )

    test_eager = df.select(
        language=detect_lang(
            "text",
            **{
                "algorithm": algorithm,
                "include_langs": include_langs,
                "exclude_langs": exclude_langs,
                "in_parallel": in_parallel,
                "low_accuracy": low_accuracy,
            },
        )
    )
    test_lazy = (
        df.lazy()
        .select(
            language=detect_lang(
                "text",
                **{
                    "algorithm": algorithm,
                    "include_langs": include_langs,
                    "exclude_langs": exclude_langs,
                    "in_parallel": in_parallel,
                    "low_accuracy": low_accuracy,
                },
            ),
        )
        .collect()
    )

    exp = pl.DataFrame(
        [
            ("French", "Russian", "English", "Japanese", None),
        ],
        schema={"language": pl.Utf8},
    )

    assert_frame_equal(test_eager, exp, check_dtype=False)
    assert_frame_equal(test_lazy, exp, check_dtype=False)


@pytest.mark.parametrize(
    """algorithm, include_langs, exclude_langs, low_accuracy""",
    [
        ("what_lang", [], [], False),
        ("what_lang", ["french"], [], False),
        ("what_lang", [], ["spanish"], False),
        ("lingua", [], [], False),
        ("lingua", [], [], True),
        # TODO: add stricter error handling in rust code for only needing 2 lang options to choose from
        ("lingua", ["french", "spanish"], [], False),
        ("lingua", ["french", "spanish"], [], True),
        ("lingua", [], ["spanish"], False),
        ("lingua", [], ["spanish"], False),
    ],
)
def test_detect_lang_confidence(
    algorithm: str,
    include_langs: list,
    exclude_langs: list,
    low_accuracy: bool,
):
    pl.Config.set_verbose(True)
    df = pl.DataFrame(
        [
            (
                "Après une enfance difficile, il est élève d'une école d'officiers et se lie avec le mouvement progressiste de Saint-Pétersbourg.",
                "23490u35",
                None,
            ),
        ],
        schema={"text": pl.Utf8},
    )

    with pl.Config(verbose=True):
        test_eager = df.select(
            confidence=detect_lang_confidence(
                "text",
                **{
                    "algorithm": algorithm,
                    "include_langs": include_langs,
                    "exclude_langs": exclude_langs,
                    "low_accuracy": low_accuracy,
                },
            ),
        )
        test_lazy = (
            df.lazy()
            .select(
                confidence=detect_lang_confidence(
                    "text",
                    **{
                        "algorithm": algorithm,
                        "include_langs": include_langs,
                        "exclude_langs": exclude_langs,
                        "low_accuracy": low_accuracy,
                    },
                ),
            )
            .collect()
        )

    assert test_eager.schema == OrderedDict([("confidence", pl.Float64)])
    assert test_lazy.schema == OrderedDict([("confidence", pl.Float64)])
