import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import json
from typing import Literal, Sequence
import matplotlib.patheffects as pe


def tornado_bucket_mean_over_time(
    *,
    scope: Literal["raw", "refined", "overall"] = "refined",
    years: Sequence[int] | None = None,        
    min_share: float = 0.001,                  # parameter to keep limit overestimation from insignificant trade
    iso2color_path: str | Path | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Tornado chart (mirrored horizontal bars, one y-row per bucket):
      • Left  = CHN→USA (mean over years of per-year bucket means)
      • Right = USA→CHN (mean over years of per-year bucket means)

    Leverage for a code = (num_share / den_share), where:
      - num_share: importer's share of imports (by QTY) from the exporter
      - den_share: exporter's share of exports (by VALUE) to the importer
    Filter: both shares >= min_share.

    Requires global dfs dict: dfs[year] is a Polars DataFrame with columns
    ["importer","exporter","cmdCode","qty","primaryValue"].
    """

    # ------------------ Buckets (ordered 10) ------------------
    CO_OVER = {260500, 282200, 282739, 283329, 283699, 291529, 810520, 810530, 810590}
    CO_RAW  = {260500};                    CO_REF  = CO_OVER - CO_RAW

    B_OVER  = {252800,281000,281290,282690,283990,284011,284019,284020,284520,284990,285000}
    B_RAW   = {252800};                    B_REF   = B_OVER - B_RAW

    GE_OVER = {261790,281219,281290,282560,285000,811292}
    GE_RAW  = {261790};                    GE_REF  = GE_OVER - GE_RAW

    IR_OVER = {261690,284390,381512,711019,711041,711049,711292}
    IR_RAW  = {261690};                    IR_REF  = IR_OVER - IR_RAW

    P_OVER  = {
        251010,251020,280470,280910,280920,281212,281213,281214,281390,
        283510,283522,283524,283525,283529,283531,283539,285390,291990,
        310319,310530,740500
    }
    P_RAW   = {251010,251020};             P_REF   = P_OVER - P_RAW

    SI_OVER = {250510,280461,280469,281122,283911,283990,284920,285000,293190,391000,720221}
    SI_RAW  = {250510};                    SI_REF  = SI_OVER - SI_RAW

    GA_OVER = {260600,260800,281219,282590,283329,283429,285000,285390,381800,811292,811299}
    GA_RAW  = {260600,260800};             GA_REF  = GA_OVER - GA_RAW

    REE_OVER= {261220,261790,280530,284690}
    REE_RAW = {261220,261790};             REE_REF = {280530,284690}

    CE_OVER = {261790,280530,284610,360690}
    CE_RAW  = {261790};                    CE_REF  = {280530,284610,360690}

    TA_OVER = {261590,282590,284990,285000,810320,810330,810391,810399}
    TA_RAW  = {261590};                    TA_REF  = TA_OVER - TA_RAW

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

    if scope not in {"raw","refined","overall"}:
        raise ValueError("scope must be one of {'raw','refined','overall'}")

    def pick_codes(over, raw, ref):
        return {"overall": over, "raw": raw, "refined": ref}[scope]

    # standard color mapping from json file
    def _load_iso2color(path: Path) -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _mpl_color(c):
        if isinstance(c, str):
            return c
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            r, g, b = c[:3]
            if max(r, g, b) > 1:
                r, g, b = r/255, g/255, b/255
            return (r, g, b) if len(c) < 4 else (r, g, b, float(c[3]))
        return None

    if iso2color_path is None:
        local   = Path.cwd() / "iso2color.json"
        default = Path("/project/bii_nssac/people/anil/DPI/iso2color.json")
        iso2color_path = local if local.exists() else (default if default.exists() else None)

    raw_map = _load_iso2color(Path(iso2color_path)) if iso2color_path else {}
    COL_USA = _mpl_color(raw_map.get("USA")) or "#1E3D8F"
    COL_CHN = _mpl_color(raw_map.get("CHN")) or "#B33A2B"

    COL_EDGE = "#111827"
    COL_GRID = "#111827"

    # check if data is present in workspace
    if "dfs" not in globals() or not isinstance(dfs, dict):
        raise RuntimeError("Missing global dfs dict in workspace.")
    all_years = sorted(dfs.keys())
    years = list(years) if years is not None else all_years
    if not years:
        raise ValueError("No years provided (and dfs appears empty).")

    ### helper functions
    def _frame_for_year(yr: int) -> pl.DataFrame:
        return (
            dfs[yr]
            .filter(
                (pl.col("exporter") != "W00") &
                (pl.col("importer") != "W00") &
                (pl.col("exporter") != pl.col("importer"))
            )
            .select(["importer","exporter","cmdCode","qty","primaryValue"])
        )

    def _leverage_for_code(df: pl.DataFrame, importer: str, exporter: str, code: int) -> float:
        cdf = df.filter(pl.col("cmdCode") == code)
        if cdf.is_empty():
            return np.nan

        imp = cdf.filter(pl.col("importer") == importer)
        if imp.is_empty():
            return np.nan
        total_imp_qty = float(imp["qty"].sum() or 0.0)
        num_qty = float(imp.filter(pl.col("exporter") == exporter)["qty"].sum() or 0.0) if total_imp_qty > 0 else 0.0
        num_share = (num_qty / total_imp_qty) if total_imp_qty > 0 else np.nan

        exp = cdf.filter(pl.col("exporter") == exporter)
        if exp.is_empty():
            return np.nan
        total_exp_val = float(exp["primaryValue"].sum() or 0.0)
        den_val = float(exp.filter(pl.col("importer") == importer)["primaryValue"].sum() or 0.0) if total_exp_val > 0 else 0.0
        den_share = (den_val / total_exp_val) if total_exp_val > 0 else np.nan

        if (np.isnan(num_share) or np.isnan(den_share) or den_share == 0
            or num_share < min_share or den_share < min_share):
            return np.nan

        return num_share / den_share

    def _bucket_mean_for_year(yr: int, codes: set[int], importer: str, exporter: str) -> float:
        df = _frame_for_year(yr)
        vals = [_leverage_for_code(df, importer, exporter, c) for c in codes]
        vals = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        return float(np.mean(vals)) if vals.size else np.nan

    # computes mean of yearly bucket mean
    names, left_vals, right_vals = [], [], []
    for name, over, raw, ref in BUCKETS:
        codes = pick_codes(over, raw, ref)
        if not codes:
            names.append(name); left_vals.append(0.0); right_vals.append(0.0); continue

        per_year_left  = [_bucket_mean_for_year(yr, codes, "USA", "CHN") for yr in years]  # CHN→USA
        per_year_right = [_bucket_mean_for_year(yr, codes, "CHN", "USA") for yr in years]  # USA→CHN

        left_mean  = float(np.nanmean(per_year_left))  if np.any(~np.isnan(per_year_left))  else 0.0
        right_mean = float(np.nanmean(per_year_right)) if np.any(~np.isnan(per_year_right)) else 0.0

        names.append(name)
        left_vals.append(left_mean)
        right_vals.append(right_mean)

    ### plot formatting
    y = np.arange(len(names))
    h = 0.72
    fig, ax = plt.subplots(figsize=(13.5, 8.2))

    TITLE_FS     = 28
    TICK_FS      = 18
    DATALABEL_FS = 18

    # tick labels
    TICK_FS = int(round(TICK_FS * 1.20))

    # bars
    ax.barh(
        y, [-v for v in left_vals],
        height=h, color=COL_CHN, edgecolor=COL_EDGE, linewidth=1.2, alpha=0.92, zorder=2
    )
    ax.barh(
        y, right_vals,
        height=h, color=COL_USA, edgecolor=COL_EDGE, linewidth=1.2, alpha=0.92, zorder=2
    )

    # center line, and force x axis labels on both axes to be positive for increased understandibility and logic
    ax.axvline(0, color=COL_EDGE, lw=2.0, zorder=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=TICK_FS, fontweight="bold", color="black")
    ax.tick_params(axis="x", labelsize=TICK_FS, colors="black", width=1.4, length=7)

    max_abs = max([0.0] + [abs(v) for v in left_vals + right_vals])
    pad = max_abs * 0.22 if max_abs > 0 else 0.6
    ax.set_xlim(-(max_abs + pad), (max_abs + pad))

    scope_short = "Raw" if scope == "raw" else ("Refined" if scope == "refined" else "Overall")
    ax.set_title(f"CHN vs. USA Leverage - {scope_short}", fontsize=TITLE_FS, fontweight="black", color="black", pad=14)

    ax.set_xlabel("Mean leverage", fontsize=26, fontweight="black", color="black", labelpad=12)
    ax.set_ylabel("Bucket", fontsize=26, fontweight="black", color="black", labelpad=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color("black")

    # gridlines off
    ax.grid(False)

    # data labels
    label_outline = [pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()]
    for i, v in enumerate(left_vals):
        if v > 0:
            ax.text(
                -(v + pad * 0.04), y[i], f"{v:.2f}",
                va="center", ha="right",
                color="black", fontsize=DATALABEL_FS, fontweight="black",
                path_effects=label_outline, zorder=4
            )
    for i, v in enumerate(right_vals):
        if v > 0:
            ax.text(
                v + pad * 0.04, y[i], f"{v:.2f}",
                va="center", ha="left",
                color="black", fontsize=DATALABEL_FS, fontweight="black",
                path_effects=label_outline, zorder=4
            )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=260, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)