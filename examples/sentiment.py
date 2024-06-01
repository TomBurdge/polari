import polari
import duckdb
from time import time
from polars import Config

# there will be 250 of EACH dataset
rows = 250
total_rows: str = "{:,}".format(rows*3)
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
print("""
    Got the dataset!

    Let's time how long it takes to detect the sentiment, just for fun.

      """)
ts = time()
df = lf.select(
    "text",
    polari.get_sentiment("text", output_type="compound").alias("sentiment"),
    polari.get_sentiment("text", output_type="pos").alias("pos"),
    polari.get_sentiment("text", output_type="neu").alias("neu"),
    polari.get_sentiment("text", output_type="neg").alias("neg"),
    "rating",
).collect()
te = time()
total_time = te - ts

Config(fmt_str_lengths=100).set_tbl_hide_dataframe_shape(True)

print(f"""
Processing {total_rows} rows took {total_time} seconds.

Here's the head of the df:

{df.head(10)}

""")
