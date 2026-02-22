# Trade Leverage & Disruption Visualization Toolkit

A small set of Python utilities for analyzing bilateral trade dependence and “disruption” patterns across commodity buckets. The toolkit produces:

- **Mean leverage tornado charts** (USA↔CHN or coalition vs coalition)
- **Top-k exporter union tables** (unique exporters appearing in top-k across years)
- **Stacked tornado charts** based on top-k union results
- **Exporter × bucket heatmaps** summarizing top-k frequency

This repo is meant to be used from a notebook or analysis script where you already have a `dfs` dictionary of Polars DataFrames loaded in memory.

---

## Contents

- `mean_leverage_tornado.py` — baseline USA↔CHN mean leverage tornado by bucket
- `coalition_mean_leverage_tornado.py` — coalition vs coalition mean leverage tornado by bucket
- `union_tables.py` — builds union tables of unique exporters in top-k disruption sets
- `union_stacked_tornado.py` — stacked tornado visualization powered by `union_tables.py`
- `unique_k_heatmap.py` — exporter × bucket frequency computation + heatmap plotting
- `iso2color.json` — ISO3 → color mapping used in plots

---

## Dependencies

Minimum:
- Python 3.10+ (3.9+ usually fine)
- `polars`
- `numpy`
- `matplotlib`

Optional / commonly used:
- `pandas`
- `pathlib`, `json` (standard library)

---

## Expected Data Format

Most functions assume you have a global variable named `dfs` in scope:

```python
dfs: dict[int, polars.DataFrame]
```

Each `dfs[year]` should contain (at minimum):

| Column         | Meaning |
|----------------|---------|
| `importer`     | ISO-3 importer |
| `exporter`     | ISO-3 exporter |
| `cmdCode`      | HS code (typically HS-6, int or str castable) |
| `qty`          | quantity (numeric) |
| `primaryValue` | trade value (numeric) |

Typical assumptions across scripts:
- Rows with `exporter == importer` may be filtered out.
- `W00` may be excluded where present.
- Years are commonly 2017–2025, but any year keys are allowed.

---

## Color Mapping (`iso2color.json`)

Several plots optionally color countries using `iso2color.json` (ISO-3 keys).

Recommended: pass the path explicitly:
```python
iso2color_path="iso2color.json"
```


---

## Key Concepts

### Commodity “Buckets”
The scripts assume commodity codes are grouped into a fixed set of buckets (e.g., REE, Ga, Ge, Ta, B, Co, Ir, P, Si, Ce). The mapping from HS codes → buckets is implemented inside the scripts.

### Leverage (high-level)
Leverage is computed from two shares (exact computation is in each script’s docstrings):

- A share of the **importer’s imports** from an exporter (often using `qty`)
- A share of the **exporter’s exports** to an importer (often using `primaryValue`)

A common form:
```
leverage = num_share / den_share
```

A `min_share` threshold is typically used to avoid noisy ratios from tiny flows.

### Top-k “Disruption Exporters”
Union/heatmap utilities identify which exporters appear in the importer’s **top-k suppliers** (k=1..3) for each HS code and/or bucket across multiple years.

---

## Usage

### 1) Mean leverage tornado (USA ↔ CHN)
File: `mean_leverage_tornado.py`  
Main function: `tornado_bucket_mean_over_time(...)`

Example:
```python
from mean_leverage_tornado import tornado_bucket_mean_over_time

# assumes dfs is defined in your notebook
tornado_bucket_mean_over_time(
    scope="refined",                 # "raw" | "refined" | "overall"
    years=range(2017, 2026),
    min_share=0.001,
    iso2color_path="iso2color.json",
    save_path="mean_leverage_tornado_refined.png",
    show=True,
)
```

Output: a mirrored tornado chart by bucket.

---

### 2) Coalition vs coalition mean leverage tornado
File: `coalition_mean_leverage_tornado.py`  
Main function: `tornado_bucket_mean_over_time_sets(...)`

Example:
```python
from coalition_mean_leverage_tornado import tornado_bucket_mean_over_time_sets

G7 = ("USA","CAN","GBR","FRA","DEU","ITA","JPN")
CHN = ("CHN",)

tornado_bucket_mean_over_time_sets(
    set_A=G7,                        # displayed on one side (see function docstring)
    set_B=CHN,                       # displayed on the other side
    scope="refined",
    years=range(2017, 2026),
    min_share=0.001,
    iso2color_path="iso2color.json",
    save_path="coalition_leverage_tornado.png",
    show=True,
)
```

---

### 3) Build top-k union tables
File: `union_tables.py`  
Main function: `unique_exporter_table_scoped(importer_iso, ...)`

Example:
```python
from union_tables import unique_exporter_table_scoped

tbl = unique_exporter_table_scoped(
    importer_iso="USA",
    scope="refined",
    metric="value",                  # or "qty"
    years=range(2017, 2026),
    include_individual_code_rows=False,
    include_bucket_rows=True,
    include_all_buckets_row=True,
    save_csv="usa_unique_exporters_refined.csv",
)

tbl
```

---

### 4) Stacked tornado based on union tables
File: `union_stacked_tornado.py`  
Main function: `tornado_topk_exporter_counts_by_bucket(...)`

This script *depends on* `unique_exporter_table_scoped(...)` being available at runtime.

Example:
```python
from union_tables import unique_exporter_table_scoped
from union_stacked_tornado import tornado_topk_exporter_counts_by_bucket

tornado_topk_exporter_counts_by_bucket(
    scope="refined",
    years=range(2017, 2026),
    iso2color_path="iso2color.json",
    save_path="topk_union_tornado.png",
    show=True,
)
```

---

### 5) Exporter × bucket heatmap (kSUM / top-k frequency)
File: `unique_k_heatmap.py`  
Main functions:
- `compute_exporter_bucket_frequency_kSUM(...)`
- `plot_exporter_bucket_heatmap_kSUM(...)`

Example:
```python
from unique_k_heatmap import (
    compute_exporter_bucket_frequency_kSUM,
    plot_exporter_bucket_heatmap_kSUM,
)

freq = compute_exporter_bucket_frequency_kSUM(
    importer="USA",
    years=range(2017, 2026),
    scope="refined",
    top_n=15,
)

plot_exporter_bucket_heatmap_kSUM(
    freq,
    title="USA: Exporter × Bucket Frequency (kSUM)",
    iso2color_path="iso2color.json",
    save_path="usa_bucket_heatmap.png",
    show=True,
)
```

