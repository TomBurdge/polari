# Polari 🌈

Polari can perform two purposes:
1. Detect the language of natural language text.
2. Detect the sentiment of English language text with a basic pre-trained algorithm.

Python's simplicity with rust's speed and scale. 🚀


# What's in a name?
"Polari (from Italian parlare 'to talk') is a form of slang or cant historically used in Britain."[Wikipedia](https://en.wikipedia.org/wiki/Polari)

Polari was spoken by "mostly camp gay men. They were a class of people who lived on the margins of society. Many of them broke the law - a law which is now seen... as being unfair and cruel - and so they were at risk of arrest, shaming, blackmail, and attack. They were not seen as important or interesting. Their stories were not told." [Fabulosa!: The Story of Polari, Britain's Secret Gay Language p. 10-11](https://shows.acast.com/betwixt-the-sheets/episodes/polari-the-secret-language-of-gay-men)

The `polari` library:
* Performs *language* & sentiment detection.
* Is a plugin for a library named *polar*s.
* Was, coincidentally, first released during Pride Month (June 2024).

If you have fun with this library, please consider donating to a charity which supports LGBTQIA+ folks.

Perhaps:
* [Stonewall](https://donorbox.org/support-stonewall)
* [Mermaids](https://mermaidsuk.org.uk/?form=donate)
* An organisation which supports people close to wherever you are in the world.

Pull requests with further charity & organisation suggestions are welcome.[^1]

# Language Detection 🔎

### Load the data quickly with hugging face & ducdkb
For quick setup with sample data, install the requirements in examples/example_requirements.txt
```ssh
# Linux/MacOS
python -m .venv && source .venv/bin/activate && python -m pip install polari duckdb==0.10.3 polars==0.20.30 pyarrow==16.1.0
```
Load some sample data:
```python
import polari
import duckdb
from time import time
from polars import Config, col

# On row limits below the millions, the LazyFrame setup with duckdb will take most of the time.
rows = 5

# here are the languages that whichlang supports
languages = (
    # The MSA and Simplified Chinese less precise names in polari.
    "Modern Standard Arabic",
    "Simplified Chinese",
    "German",
    "English",
    "French",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Dutch",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swedish",
    "Turkish",
    "Vietnamese",
)
# set up the LazyFrame
lf = (
    duckdb.sql(
        f"SELECT inputs, language FROM 'hf://datasets/CohereForAI/aya_dataset/data/train-00000-of-00001.parquet' WHERE language in {languages} LIMIT {str(rows)};"
    )
    .pl()
    .lazy()
)
```


### Detect the language 🌐 🔎
```python
Config.set_tbl_hide_dataframe_shape(True)

df = lf.select(
    "inputs",
    polari.detect_lang("inputs", algorithm="which_lang").alias("detected_lang"),
    col("language").alias("true_lang"),
).collect()

print(df)
```

```
┌─────────────────────────────────┬───────────────┬────────────┐
│ inputs                          ┆ detected_lang ┆ true_lang  │
│ ---                             ┆ ---           ┆ ---        │
│ str                             ┆ str           ┆ str        │
╞═════════════════════════════════╪═══════════════╪════════════╡
│ Hãy tiếp tục đoạn văn sau:      ┆ Vietnamese    ┆ Vietnamese │
│ "T…                             ┆               ┆            │
│ Bu paragrafın devamını yazın: … ┆ Turkish       ┆ Turkish    │
│ ¿Cuál es la respuesta correcta… ┆ Spanish       ┆ Spanish    │
│ 中押(ちゅうお)し勝ちといえば、  ┆ Japanese      ┆ Japanese   │
│ どんなゲームの勝負の決まり方…   ┆               ┆            │
│ Em que ano os filmes deixaram … ┆ Portuguese    ┆ Portuguese │
└─────────────────────────────────┴───────────────┴────────────┘
```
### Algorithms
The above is only with the `whichlang` algorithm, the quickest and simplest algorithm.

Two of the algorithms can output a confidence score with `detect_lang_confidence`: what_lang, and lingua.

Supported algorithms:[^2]
* `what_lang`
* `lingua`
* `whichlang`

`what_lang` and `lingua` support language subsets and language exclusion.
`lingua` supports high and low accuracy mode.

### Detect the script 📜
It is also possible to detect the script of the dataset with `what_lang` and `lingua`.

```python
df = lf.select(
    "inputs",
    "language",
    polari.detect_script("inputs").alias("detected_script"),
).collect()

print(df.head(3))
```
```
┌────────────────────────────────────────────────────────┬─────────────────┬─────────────────┐
│ inputs                                                 ┆ language        ┆ detected_script │
│ ---                                                    ┆ ---             ┆ ---             │
│ str                                                    ┆ str             ┆ str             │
╞════════════════════════════════════════════════════════╪═════════════════╪═════════════════╡
│ Heestan waxaa qada Khalid Haref Ahmed                  ┆ Somali          ┆ Latin           │
│ OO ku Jiray Kooxdii Dur Dur!                           ┆                 ┆                 │
│ Quels président des États-Unis ne s’est jamais marié ? ┆ French          ┆ Latin           │
│ كم عدد الخلفاء الراشدين ؟ أجب على السؤال السابق.    ┆ Standard Arabic ┆ Arabic          │
└────────────────────────────────────────────────────────┴─────────────────┴─────────────────┘
    
```
# Sentiment Detection 😀😠
`polari` can detect the sentiment of English language text via a rust port of [VADER](https://github.com/ckw017/vader-sentiment-rust).

The pre-trained model was originally trained for sentiment detection on social media posts, but has semi-decent performance on opinionated text. The below performs analysis on amazon reviews.
### Sample Data
```python
import polari
import duckdb
from time import time
from polars import Config

# On row limits below the millions, the LazyFrame setup with duckdb will take most of the time.
# This will load {rows} of 1*, 3*, and 5* reviews.
rows = 1
subset="Beauty_and_Personal_Care"
dataset = f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{subset}.jsonl"
# set up the LazyFrame
lf = (
    duckdb.sql(
    f"""
    WITH positive as(
            SELECT text, rating FROM '{dataset}' WHERE rating = 5 LIMIT {rows}
        )
        , neutral as(
            SELECT text, rating FROM '{dataset}' WHERE rating = 3 LIMIT {rows}
        )
        , negative as(
            SELECT text, rating FROM '{dataset}' WHERE rating = 1 LIMIT {rows}
        )
    SELECT * FROM positive
    UNION ALL
    SELECT * FROM negative
    UNION ALL
    SELECT * FROM neutral;
    """
)
    .pl()
    .lazy()
)
```


### Detect Sentiment 😀😠🔎


```python
df = lf.select(
    "text",
    polari.get_sentiment("text", output_type="compound").alias("sentiment"),
    polari.get_sentiment("text", output_type="pos").alias("sentiment"),
    polari.get_sentiment("text", output_type="neu").alias("sentiment"),
    polari.get_sentiment("text", output_type="neg").alias("sentiment"),
    "rating",
).collect()

df.head()
```
```
┌──────────────────────────────────────────────────────┬───────────┬──────────┬──────────┬──────────┬────────┐
│ text                                                 ┆ sentiment ┆ pos      ┆ neu      ┆ neg      ┆ rating │
│ ---                                                  ┆ ---       ┆ ---      ┆ ---      ┆ ---      ┆ ---    │
│ str                                                  ┆ f64       ┆ f64      ┆ f64      ┆ f64      ┆ f64    │
╞══════════════════════════════════════════════════════╪═══════════╪══════════╪══════════╪══════════╪════════╡
│ Bought this for my granddaughter.  Her entire family…┆ 0.63695   ┆ 0.21875  ┆ 0.78125  ┆ 0.0      ┆ 5.0    │
│ This is a good product but it doesn't last very long…┆ 0.238227  ┆ 0.130435 ┆ 0.869565 ┆ 0.0      ┆ 3.0    │
│ Tops the list for worst purchase. Tried these for al…┆ -0.939365 ┆ 0.094854 ┆ 0.735183 ┆ 0.169963 ┆ 1.0    │
└──────────────────────────────────────────────────────┴───────────┴──────────┴──────────┴──────────┴────────┘
```
Output types include:
* compound
* neutral
* positive
* negative.


# Credits
Language detection:
* [whatlang](https://github.com/greyblake/whatlang-rs)
* [lingua](https://github.com/pemistahl/lingua-rs)
* [whichlang](https://github.com/quickwit-oss/whichlang/)

Sentiment:
* [VaderSentiment](https://github.com/cjhutto/vaderSentiment)

Polars:
* The [Polars DataFrame library](https://github.com/pola-rs/polars).
* **Marco Gorelli**. In addition to being a stalwart in the open source DataFrame community, Marco has made an **incredible** [tutorial for making polars plugins](https://marcogorelli.github.io/polars-plugins-tutorial/).

### Footnotes
[^1]: In the **extremely** unlikely scenario that this project becomes popular, and therefore a library that needs to sustain itself, users could *also* be invited to donate to the project in a separate section of the README.

[^2]: Benchmarking algorithm prediction precision/recall can be done with `polari`. Difference in detection *speed* by algorithm may be due to the implementation in `polari`, rather than the original rust crate.