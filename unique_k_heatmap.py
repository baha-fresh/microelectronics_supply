import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal, Sequence
import json
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

#List of buckets
def _bucket_sets():
    CO_OVER = {260500, 282200, 282739, 283329, 283699, 291529, 810520, 810530, 810590}
    CO_RAW  = {260500};                    CO_REF  = CO_OVER - CO_RAW

    B_OVER  = {252800,281000,281290,282690,283990,284011,284019,284020,284520,284990,285000}
    B_RAW   = {252800};                    B_REF   = B_OVER - B_RAW

    GE_OVER = {261790,281219,281290,282560,285000,811292}
    GE_RAW  = {261790};                    GE_REF  = GE_OVER - GE_RAW

    IR_OVER = {261690,284390,381512,711019,711041,711049,711292}
    IR_RAW  = {261690};                    IR_REF  = IR_OVER - IR_RAW

    P_OVER  = {251010,251020,280470,280910,280920,281212,281213,281214,281390,
               283510,283522,283524,283525,283529,283531,283539,285390,291990,
               310319,310530,740500}
    P_RAW   = {251010,251020};             P_REF   = P_OVER - P_RAW

    SI_OVER = {250510,280461,280469,281122,283911,283990,284920,285000,293190,391000,720221}
    SI_RAW  = {250510};                    SI_REF   = SI_OVER - SI_RAW

    GA_OVER = {260600,260800,281219,282590,283329,283429,285000,285390,381800,811292,811299}
    GA_RAW  = {260600,260800};             GA_REF   = GA_OVER - GA_RAW

    REE_OVER= {261220,261790,280530,284690}
    REE_RAW = {261220,261790};             REE_REF  = {280530,284690}

    CE_OVER = {261790,280530,284610,360690}
    CE_RAW  = {261790};                    CE_REF   = {280530,284610,360690}

    TA_OVER = {261590,282590,284990,285000,810320,810330,810391,810399}
    TA_RAW  = {261590};                    TA_REF   = TA_OVER - TA_RAW

    BUCKETS = [
        ("REE", REE_OVER, REE_RAW, REE_REF),
        ("Ga",  GA_OVER,  GA_RAW,  GA_REF),
        ("Ge",  GE_OVER,  GE_RAW,  GE_REF),
        ("Ta",  TA_OVER,  TA_RAW,  TA_REF),
        ("B",   B_OVER,   B_RAW,   B_REF),
        ("Co",  CO_OVER,  CO_RAW,  CO_REF),
        ("Ir",  IR_OVER,  IR_RAW,  IR_REF),
        ("P",   P_OVER,   P_RAW,   P_REF),
        ("Si",  SI_OVER,  SI_RAW,  SI_REF),
        ("Ce",  CE_OVER,  CE_RAW,  CE_REF),
    ]
    return BUCKETS

def _codes_for_scope(over, raw, ref, scope: str):
    return {"overall": over, "raw": raw, "refined": ref}[scope]

# using color mapping dictionary
def _load_iso2color(iso2color_path: str | Path | None):
    if not iso2color_path:
        local   = Path.cwd() / "iso2color.json"
        default = Path("/project/bii_nssac/people/anil/DPI/iso2color.json")
        iso2color_path = local if local.exists() else (default if default.exists() else None)
    if not iso2color_path:
        return {}
    try:
        with open(iso2color_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _mpl_color(c):
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        r, g, b = c[:3]
        if max(r, g, b) > 1:
            r, g, b = r/255.0, g/255.0, b/255.0
        if len(c) >= 4:
            a = c[3]; return (r, g, b, a)
        return (r, g, b)
    return None

def _make_gradient(base_light: str, max_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom_grad", [base_light, max_color], N=256)

# rank exporters based on total value flow to selected importer
def _top_exporters_by_flow(importer: str, years: Sequence[int], scope: str, top_n: int = 15) -> list[str]:
    BUCKETS = _bucket_sets()
    codes_union = set()
    for _, over, raw, ref in BUCKETS:
        codes_union |= _codes_for_scope(over, raw, ref, scope)

    total = Counter()
    for yr in years:
        if yr not in dfs:
            continue
        df = (
            dfs[yr]
            .filter(
                (pl.col("importer")==importer) &
                (pl.col("exporter")!="W00") &
                (pl.col("importer")!="W00") &
                (pl.col("exporter")!=pl.col("importer")) &
                (pl.col("cmdCode").is_in(list(codes_union)))
            )
            .select(["exporter","primaryValue"])
            .group_by("exporter")
            .agg(pl.col("primaryValue").sum().alias("v"))
        )
        for exp, v in df.iter_rows():
            total[exp] += float(v or 0.0)

    ranked = [e for e,_ in total.most_common(top_n)]
    return ranked

#For a (bucket,year,k): return the top-k exporters for importer
def _max_disrupt_topk_for_bucket_year(
    importer: str,
    scope: str,
    year: int,
    bucket_codes: set[int],
    k: int,
    metric_col: str,        # "primaryValue" or "qty" (for value or quantity)
) -> list[str]:
    if year not in dfs:
        return []

    base = (
        dfs[year]
        .filter(
            (pl.col("cmdCode").is_in(list(bucket_codes))) &
            (pl.col("exporter")!="W00") &
            (pl.col("importer")!="W00") &
            (pl.col("exporter")!=pl.col("importer"))
        )
        .select(["importer","exporter","cmdCode","qty","primaryValue"])
    )
    if base.is_empty():
        return []

    best_loss = -1.0
    best_topk = []
    for code in set(bucket_codes):
        cdf = base.filter(pl.col("cmdCode")==code)
        if cdf.is_empty():
            continue

        tot = float(cdf.filter(pl.col("importer")==importer)[metric_col].sum() or 0.0)
        if tot <= 0:
            continue

        rank = (
            cdf.filter(pl.col("importer")==importer)
               .group_by("exporter")
               .agg(pl.col(metric_col).sum().alias("flow"))
               .sort("flow", descending=True)
        )
        if rank.is_empty():
            continue

        top = rank.head(k)
        top_sum = float(top["flow"].sum() or 0.0)
        loss_pct = 100.0 * (top_sum / tot) if tot > 0 else 0.0

        if loss_pct > best_loss:
            best_loss = loss_pct
            best_topk = [e for e in top["exporter"].to_list()]

    return best_topk

# compute SUM across k levels per year
def compute_exporter_bucket_frequency_kSUM(
    *,
    importer: Literal["USA","CHN"],
    scope: Literal["raw","refined"]="refined",
    years: Sequence[int]=range(2017, 2026),
    k_levels: Sequence[int]=(1,2,3),
    top_exporters: int = 15,
    ranking_metric: Literal["value","qty"]="value",
):
    """
    Returns:
      M: (E × 10) matrix where each cell = SUM over (years × k_levels) of
         appearances in the max-disrupt set for that (bucket, year, k).
         So max = len(years) * len(k_levels) (e.g., 9 * 3 = 27).
      row_labels: exporter ISO codes (top by value to importer)
      col_labels: 10 bucket names
    """
    metric_col = "primaryValue" if ranking_metric=="value" else "qty"
    BUCKETS = _bucket_sets()
    bucket_names = [b[0] for b in BUCKETS]

    exporters = _top_exporters_by_flow(importer, years, scope, top_n=top_exporters)
    if not exporters:
        return np.zeros((0,0), dtype=int), [], []

    freq = Counter()   # (exporter, bucket) -> integer count

    for yr in years:
        for (bname, over, raw, ref) in BUCKETS:
            codes = _codes_for_scope(over, raw, ref, scope)
            if not codes:
                continue
            for k in k_levels:
                topk = _max_disrupt_topk_for_bucket_year(
                    importer=importer, scope=scope, year=yr,
                    bucket_codes=codes, k=k, metric_col=metric_col
                )
                for e in topk:
                    if e in exporters:
                        freq[(e, bname)] += 1  # counts each k occurence

    M = np.zeros((len(exporters), len(bucket_names)), dtype=int)
    for ei, e in enumerate(exporters):
        for bj, b in enumerate(bucket_names):
            M[ei, bj] = freq.get((e, b), 0)

    return M, exporters, bucket_names

# makes a heatmap, default is set to a gradient from light gray to the color of the country in the json file
def plot_exporter_bucket_heatmap_kSUM(
    *,
    importer: Literal["USA","CHN"],
    scope: Literal["raw","refined"]="refined",
    years: Sequence[int]=range(2017, 2026),
    k_levels: Sequence[int]=(1,2,3),
    top_exporters: int = 15,
    iso2color_path: str | Path | None = None,
    annotate: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Heatmap only: each cell is the SUM over (years × k_levels) of max-disrupt appearances.
    Base color = light gray; gradient max = USA dark blue (if importer='USA') or CHN dark red (if importer='CHN').
    """
    # Load colors
    iso_map = _load_iso2color(iso2color_path)
    COL_USA = _mpl_color(iso_map.get("USA")) or "#1E3D8F"
    COL_CHN = _mpl_color(iso_map.get("CHN")) or "#B33A2B"
    country_color = COL_USA if importer == "USA" else COL_CHN
    base_gray = "#E6E8EB"

    # Data
    M, row_labels, col_labels = compute_exporter_bucket_frequency_kSUM(
        importer=importer, scope=scope, years=years,
        k_levels=k_levels, top_exporters=top_exporters
    )
    if M.size == 0:
        print("No data to plot.")
        return

    vmax = len(list(years)) * len(list(k_levels))  # e.g., 9 * 3 = 27

    # label sizes
    y_label_fs = 22 
    x_label_fs = 22  

    # figure details and formatting
    fig, ax = plt.subplots(
        figsize=(min(18, 8 + 0.5*len(col_labels)), 0.6*len(row_labels) + 3)
    )
    cmap = _make_gradient(base_gray, country_color)

    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=y_label_fs, fontweight="bold")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=x_label_fs, fontweight="bold")

    
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)

    ax.set_title(
        f"{importer} Disruption Heatmap - {scope} materials",
        fontsize=26, fontweight="black", pad=14
    )

    # color bar on the right side to show gradient scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    ticks = np.arange(0, vmax + 1, 9)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=22, colors="black")
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontweight("black")

    # numbers on the cells formatting
    if annotate and len(row_labels) <= 30:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = int(M[i, j])
                if v > 0:
                    txt_color = "white" if v >= (0.5 * vmax) else "black"
                    outline = "black" if txt_color == "white" else "white"
                    ax.text(
                        j, i, str(v),
                        va="center", ha="center",
                        fontsize=25,
                        fontweight="black",
                        color=txt_color,
                        path_effects=[
                            pe.Stroke(linewidth=3.5, foreground=outline),
                            pe.Normal()
                        ]
                    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)