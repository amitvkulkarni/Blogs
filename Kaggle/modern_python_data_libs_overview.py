# =============================================================================
# SECTION 1: Pandas vs Polars — GroupBy Performance Benchmark
# =============================================================================
# Compares the time (in seconds) to perform a grouped aggregation on 200,000
# rows using pandas and Polars. timeit runs each statement 10 times and
# returns the total elapsed time in seconds.
# =============================================================================

import timeit

setup = """
import pandas as pd, polars as pl
import numpy as np
n = 200_000
df_pd = pd.DataFrame({'g': np.random.randint(0, 100, n), 'v': np.random.rand(n)})
df_pl = pl.DataFrame(df_pd)  # convert pandas DataFrame to Polars
"""

stmt_pd = "df_pd.groupby('g')['v'].sum()"
stmt_pl = "df_pl.group_by('g').agg(pl.col('v').sum())"

# Output: total seconds for 10 runs (divide by 10 for per-run average)
print("pandas:", timeit.timeit(stmt_pd, setup=setup, number=10))
print("polars:", timeit.timeit(stmt_pl, setup=setup, number=10))


# =============================================================================
# SECTION 2: Ibis — Backend-Agnostic DataFrame Transformations
# =============================================================================
# Ibis lets you write transformation logic once and swap the execution backend
# (DuckDB, Polars, Postgres, etc.) by changing a single connection line.
# The query expression is compiled and sent to the backend on .execute().
# =============================================================================

import ibis

# Switch backends by changing only this connection line — the logic below stays identical:
# con = ibis.duckdb.connect()       # local files / in-process analytics
# con = ibis.polars.connect()       # Polars-speed in-memory processing
con = ibis.postgres.connect(host=...)  # production Postgres database

table = con.table("sales_data")

result = (
    table.filter(table.revenue > 1000)  # keep rows where revenue > 1000
    .group_by("region")
    .aggregate(total=table.revenue.sum())  # sum revenue per region
    .execute()  # sends the optimised query to the backend
)


# =============================================================================
# SECTION 3: DuckDB + Polars — In-Process SQL on a Polars DataFrame
# =============================================================================
# DuckDB can query a Polars DataFrame directly in memory (zero-copy) using
# standard SQL. Results can be returned as a Polars DataFrame via .pl().
# =============================================================================

import duckdb
import polars as pl

# Create sample data as a Polars DataFrame
dummy_data = pl.DataFrame(
    {
        "region": ["North", "South", "East", "West", "North", "East"],
        "revenue": [1500, 2500, 1200, 3000, 1100, 2200],
        "year": [2023, 2025, 2026, 2024, 2025, 2027],
        "column_a": [5, 12, 8, 15, 3, 20],
    }
)

# Example 1: Grouped aggregation — DuckDB registers 'dummy_data' automatically
results = duckdb.sql(
    """
    SELECT region, SUM(revenue) AS total_rev
    FROM dummy_data
    WHERE year > 2024
    GROUP BY region
    """
).pl()  # return result as a Polars DataFrame

print("Aggregated Results (back in Polars):")
print(results)

# Example 2: Simple filter — print via DuckDB's built-in .show()
print("\nFiltered Rows where column_a > 10:")
duckdb.sql("SELECT * FROM dummy_data WHERE column_a > 10").show()


# =============================================================================
# SECTION 4: Rich — Pretty Printing Dicts and Styled Tables in the Terminal
# =============================================================================
# Rich automatically applies syntax highlighting to Python objects and supports
# building formatted, coloured tables with minimal code.
# =============================================================================

from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Example 1: Pretty-print a dictionary with automatic colour highlighting
data = {"status": "success", "results": [1, 2, 3], "meta": {"version": "2.0"}}
rprint(data)  # colours keys, values, and nested structures automatically

# Example 2: Build a styled table
table = Table(title="Top Data Libraries 2026")
table.add_column("Library", style="cyan", no_wrap=True)
table.add_column("Purpose", style="magenta")
table.add_column("Speed Score", justify="right", style="green")

table.add_row("Polars", "DataFrames", "9.5")
table.add_row("DuckDB", "Analytical SQL", "9.8")
table.add_row("Rich", "CLI UI", "N/A")

console = Console()
console.print(table)


# =============================================================================
# SECTION 5: Rich — Styled Output, Progress Bars, and Live Updates
# =============================================================================
# Rich supports inline markup for styled text, progress tracking with a
# spinner/bar, and a Live context manager for real-time terminal dashboards.
# =============================================================================

from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.live import Live
import time

console = Console()

# Styled inline text using Rich markup tags
console.print("[bold green]Processed[/] 42 rows in [yellow]0.12s[/]")

# Simple two-column status table
status_table = Table()
status_table.add_column("step")
status_table.add_column("status")
status_table.add_row("ingest", "done")
status_table.add_row("transform", "running")
console.print(status_table)

# Progress bar — wraps any iterable; displays a live bar with description
for i in track(range(500), description="syncing"):
    time.sleep(0.01)

# Live context manager — useful for lightweight real-time terminal dashboards
with Live(console=console, refresh_per_second=4) as live:
    for i in range(10):
        live.update(f"Current: [bold]{i}[/]")
        time.sleep(0.2)


# =============================================================================
# SECTION 6: orjson — Fast JSON Serialisation with Dataclass Support
# =============================================================================
# orjson is a high-performance JSON library that natively serialises Python
# dataclasses, numpy arrays, and datetime objects. It returns/accepts bytes
# rather than str, making it efficient for network and file I/O.
# =============================================================================

from dataclasses import dataclass
import orjson


@dataclass
class Point:
    x: float
    y: float


p = Point(1.0, 2.0)

# Serialise to bytes; OPT_SERIALIZE_DATACLASS handles the dataclass,
# OPT_SORT_KEYS produces a canonical (reproducible) key order.
b = orjson.dumps(p, option=orjson.OPT_SERIALIZE_DATACLASS | orjson.OPT_SORT_KEYS)

obj = orjson.loads(b)  # deserialise bytes back to a Python dict

print(b)  # b'{"x":1.0,"y":2.0}'
print(obj)  # {'x': 1.0, 'y': 2.0}
