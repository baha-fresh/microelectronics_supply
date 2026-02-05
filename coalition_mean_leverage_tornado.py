from pathlib import Path
import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

def tornado_bucket_mean_over_time_sets(
    *,
    set_A=("USA",),        # set A is on the right of the plot
    set_B=("CHN",),
    scope="refined",       # standard classifications of: "raw","refined","overall"
    years=None,            # all years in scraped data (2017-2025)
    min_share=0.001,       # forces a minimum trade share for numerator and denominator in leverage ratio definition to ignore insignificant trade
    iso2color_path=None,
    show=True,
    save_path=None,
):
    """
    Tornado of MEAN 'reconfigured leverage' by bucket over a time window, with SETS.

    Updates per your request:
      • Stripe overlay = HORIZONTAL stripes (back from vertical)
      • Title: remove "Leverage of" and remove "materials"
      • Increase font sizes of everything EXCEPT title (title stays same size)
    """
    # ---- local imports so the function works even if the notebook cell didn't import typing ----

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

    if scope not in {"raw", "refined", "overall"}:
        raise ValueError("scope must be one of {'raw','refined','overall'}")

    def pick_codes(over, raw, ref):
        return {"overall": over, "raw": raw, "refined": ref}[scope]

    # ------------------ Colors (json) ------------------
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

    def _color_for_iso(iso: str, fallback: str = "#9CA3AF"):
        return _mpl_color(raw_map.get(iso.upper())) or fallback

    # Higher-contrast styling
    COL_BASE = "#ECEFF3"
    COL_EDGE = "#111827"
    COL_GRID = "#111827"

    # ------------------ Years & Data ------------------
    if "dfs" not in globals():
        raise RuntimeError("Missing `dfs` dict in workspace.")
    all_years = sorted(dfs.keys())
    years = list(years) if years is not None else all_years

    # Normalize sets, ensure USA on the RIGHT (Set A)
    A = {str(s).upper() for s in set_A}
    B = {str(s).upper() for s in set_B}
    overlap = A & B
    if overlap:
        A -= overlap
        B -= overlap
    if "USA" in B and "USA" not in A:
        A, B = B, A

    A_colors = [_color_for_iso(iso) for iso in sorted(A)]
    B_colors = [_color_for_iso(iso) for iso in sorted(B)]
    if not A_colors:
        A_colors = ["#1E3D8F"]
    if not B_colors:
        B_colors = ["#B33A2B"]

    # ------------------ Helpers ------------------
    def _frame_for_year(yr: int) -> pl.DataFrame:
        return (
            dfs[yr]
            .filter(
                (pl.col("exporter") != "W00") &
                (pl.col("importer") != "W00") &
                (pl.col("exporter") != pl.col("importer"))
            )
            .select(["importer", "exporter", "cmdCode", "qty", "primaryValue"])
        )

    def _leverage_set_over_set(df: pl.DataFrame, X: set[str], Y: set[str], code: int) -> float:
        cdf = df.filter(pl.col("cmdCode") == code)
        if cdf.is_empty():
            return np.nan

        # remove within-set flows
        cdf = cdf.filter(
            ~(
                ((pl.col("exporter").is_in(list(X))) & (pl.col("importer").is_in(list(X)))) |
                ((pl.col("exporter").is_in(list(Y))) & (pl.col("importer").is_in(list(Y))))
            )
        )
        if cdf.is_empty():
            return np.nan

        # numerator: into Y, by qty share from X
        into_Y = cdf.filter(pl.col("importer").is_in(list(Y)))
        if into_Y.is_empty():
            return np.nan
        total_imp_qty = float(into_Y.filter(~pl.col("exporter").is_in(list(Y)))["qty"].sum())
        from_X_to_Y_qty = float(into_Y.filter(pl.col("exporter").is_in(list(X)))["qty"].sum())
        num_share = (from_X_to_Y_qty / total_imp_qty) if total_imp_qty > 0 else np.nan

        # denominator: from X, by value share to Y
        from_X = cdf.filter(pl.col("exporter").is_in(list(X)))
        if from_X.is_empty():
            return np.nan
        total_exp_val = float(from_X.filter(~pl.col("importer").is_in(list(X)))["primaryValue"].sum())
        to_Y_val = float(from_X.filter(pl.col("importer").is_in(list(Y)))["primaryValue"].sum())
        den_share = (to_Y_val / total_exp_val) if total_exp_val > 0 else np.nan

        if (
            np.isnan(num_share) or np.isnan(den_share) or den_share == 0
            or num_share < min_share or den_share < min_share
        ):
            return np.nan
        return num_share / den_share

    def _bucket_mean_for_year(yr: int, codes: set[int], X: set[str], Y: set[str]) -> float:
        df = _frame_for_year(yr)
        vals = [_leverage_set_over_set(df, X, Y, c) for c in codes]
        vals = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        return float(np.mean(vals)) if vals.size else np.nan

    # ------------------ Compute means ------------------
    names, left_vals, right_vals = [], [], []
    for name, over, raw, ref in BUCKETS:
        codes = pick_codes(over, raw, ref)
        if not codes:
            names.append(name)
            left_vals.append(0.0)
            right_vals.append(0.0)
            continue

        per_year_A_over_B = [_bucket_mean_for_year(yr, codes, A, B) for yr in years]
        per_year_B_over_A = [_bucket_mean_for_year(yr, codes, B, A) for yr in years]

        A_mean = float(np.nanmean(per_year_A_over_B)) if np.any(~np.isnan(per_year_A_over_B)) else 0.0
        B_mean = float(np.nanmean(per_year_B_over_A)) if np.any(~np.isnan(per_year_B_over_A)) else 0.0

        names.append(name)
        right_vals.append(A_mean)  # A over B (RIGHT)
        left_vals.append(B_mean)   # B over A (LEFT)

    # ------------------ Plot ------------------
    y = np.arange(len(names))
    h = 0.72
    fig, ax = plt.subplots(figsize=(13.5, 8.2))

    # Title stays same
    TITLE_FS = 30

    # Everything else bigger
    AXIS_LABEL_FS = 24   # was 20
    TICK_FS       = 22   # was 18
    DATALABEL_FS  = 24   # was 18

    # Base bars
    ax.barh(y, [-v for v in left_vals], height=h, color=COL_BASE, edgecolor=COL_EDGE, linewidth=1.6, zorder=1)
    ax.barh(y, right_vals,            height=h, color=COL_BASE, edgecolor=COL_EDGE, linewidth=1.6, zorder=1)

    # ---- HORIZONTAL STRIPES across bar height (back from vertical) ----
    def _stripe_overlay_horizontal(yc: float, width: float, height: float, colors: list, alpha: float = 0.95):
        if width == 0 or not colors:
            return
        W = abs(width)
        if W <= 0:
            return

        n = max(1, len(colors))
        stripe_h = height / n
        y0 = yc - height / 2.0

        # bar spans x in [0,width] if width>0 else [width,0]
        x0 = min(0.0, width)
        x1 = max(0.0, width)

        for i in range(n):
            yy = y0 + i * stripe_h
            ax.add_patch(
                Rectangle(
                    (x0, yy),
                    x1 - x0,
                    stripe_h,
                    facecolor=colors[i % n],
                    edgecolor="none",
                    alpha=alpha,
                    zorder=2,
                )
            )

    for i, v in enumerate(left_vals):
        if v > 0:
            _stripe_overlay_horizontal(yc=y[i], width=-v, height=h, colors=B_colors)
    for i, v in enumerate(right_vals):
        if v > 0:
            _stripe_overlay_horizontal(yc=y[i], width= v, height=h, colors=A_colors)

    # Center line + ticks (this is the only "gridline" left)
    ax.axvline(0, color=COL_EDGE, lw=2.2, zorder=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=TICK_FS, fontweight="bold", color="black")
    ax.tick_params(axis="x", labelsize=TICK_FS, colors="black", width=1.6, length=8)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")

    max_abs = max([0.0] + [abs(v) for v in left_vals + right_vals])
    pad = max_abs * 0.22 if max_abs > 0 else 0.6
    ax.set_xlim(-(max_abs + pad), (max_abs + pad))

    def _set_label(S: set[str]) -> str:
        COALITION = {"CAN", "USA", "AUS", "DEU"}
        if set(S) == COALITION:
            return "Coalition"
        s = sorted(S)
        if not s:
            return "∅"
        return s[0] if len(s) == 1 else " + ".join(s)

    scope_word = {"overall": "Overall", "raw": "Raw", "refined": "Refined"}[scope]
    ax.set_title(
        f"{_set_label(B)} Leverage Over {_set_label(A)} - {scope_word}",
        fontsize=TITLE_FS, fontweight="black", color="black", pad=14
    )

    ax.set_xlabel("Mean leverage", fontsize=AXIS_LABEL_FS, fontweight="black", color="black", labelpad=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

    # ✅ Remove ALL gridlines (keep only the center line drawn above)
    ax.grid(False)

    # Labels with outline (bigger)
    label_outline = [pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()]
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