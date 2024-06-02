import polari
from time import time
from polars import Config, col
from polari.examples.datasets import load_aya

rows = 5
rows_str = "{:,}".format(rows)

# here are the languages that whichlang supports
languages = (
    # changed MSA and Simplified Chinese to their less precise lang names for the API.
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
lf = load_aya(limit=rows, language_subset=languages)

print(f"""
    Got the dataset!

    Let's time how long it takes to detect the language for {rows_str} rows, just for fun...

      """)
ts = time()
df = lf.select(
    "inputs",
    polari.detect_lang("inputs", algorithm="which_lang").alias("detected_lang"),
    col("language").alias("true_lang"),
).collect()
te = time()
total_time = te - ts

Config.set_tbl_hide_dataframe_shape(True)

print(f"""
Processing {rows_str} rows took {total_time} seconds.

Here's the head of the df:

{df.head(10)}

This is only with whichlang, the quickest and simplest algorithm. Try it out with the others!
""")
