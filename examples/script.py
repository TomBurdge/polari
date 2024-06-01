import polari
from time import time
from polars import Config
from examples.utils import load_aya

rows = 5_000
rows_str = "{:,}".format(rows)

# set up the LazyFrame
lf = load_aya(limit=rows, language_subset=None)

print(f"""
    Got the dataset!

    Let's time how long it takes to detect the script for {rows_str} rows, just for fun...
    """)
ts = time()
df = lf.select(
    "inputs",
    "language",
    polari.detect_script("inputs").alias("detected_script"),
).collect()
te = time()
total_time = te - ts

Config(fmt_str_lengths=100).set_tbl_hide_dataframe_shape(True)
print(f"""
Processing {rows_str} rows took {total_time} seconds.

Here's the head of the df:

{df.head(10)}
""")
