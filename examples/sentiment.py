import polari
import duckdb
from time import time
from polars import Config

rows = 50
# set up the LazyFrame
lf = (
    duckdb.sql(
        f"SELECT text, rating FROM 'hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Sports_and_Outdoors.jsonl' LIMIT {rows};"
    )
    .pl()
    .lazy()
)
print("""
    Got the dataset!

    Let's time how long it takes to detect the sentiment, just for fun.

      """)
ts = time()
df = lf.select(
    "text",
    polari.get_sentiment("text", output_type="compound").alias("sentiment"),
    "rating",
).collect()
te = time()
total_time = te - ts

Config.set_tbl_hide_dataframe_shape(True)
with Config(fmt_str_lengths=100):
    print(f"""
    Processing {rows} rows took {total_time} seconds.

    Here's the head of the df:

{df.head()}

    """)
