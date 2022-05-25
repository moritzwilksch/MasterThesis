from rich.panel import Panel
import polars as pl
from rich import print
from rich.console import Console
c = Console()

df = pl.read_parquet("data/raw/text_only.parquet")

while True:
    c.clear()
    sample = df.select("text").sample(5).rows()
    for s in sample:
        p = Panel(s[0])
        c.print(p)
    input()
