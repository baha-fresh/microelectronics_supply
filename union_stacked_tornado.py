import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import json

def tornado_topk_exporter_counts_by_bucket(
    *,
    scope: str = "refined",                 #different options are: "raw" | "refined" | "overall"
    years=range(2017, 2025+1),              
    iso2color_path: str | Path | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Tornado plot of occurrences in max-disruption exporter sets per bucket
    (k=1,2,3 on each side; Left=CHN, Right=USA).

    Updates:
      • Remove ALL gridlines (no vertical gridlines)
      • Keep centerline at x=0
      • Distinguish k=1,k=2,k=3 via shade/intensity (alpha) while keeping country hue
      • Keep 40% larger fonts + existing layout
      • Remove the word "disruptor" from the title
      • Bars remain solid (no hatch)
    """

    # builds union table first before it gets converted into a plot
    _fn = globals().get("unique_exporter_table_scoped")
    if not callable(_fn):
        raise RuntimeError("unique_exporter_table_scoped(...) is not defined in this notebook.")

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
            if max(r, g, b) > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            if len(c) >= 4:
                a = c[3]
                return (r, g, b, a)
            return (r, g, b)
        return None

    if iso2color_path is None:
        local   = Path.cwd() / "iso2color.json"
        default = Path("/project/bii_nssac/people/anil/DPI/iso2color.json")
        iso2color_path = local if local.exists() else (default if default.exists() else None)

    raw_map = _load_iso2color(Path(iso2color_path)) if iso2color_path else {}
    COL_USA = _mpl_color(raw_map.get("USA")) or "#1E3D8F"
    COL_CHN = _mpl_color(raw_map.get("CHN")) or "#B33A2B"

    # different shades for different k-levels. Dark to light as k increases
    A1, A2, A3 = 1.00, 0.70, 0.45

    # bucket level rows are pulled from the data
    usa_tbl = _fn(
        importer_iso="USA",
        scope=scope,
        metric="value",
        years=years,
        include_individual_code_rows=False,
        include_bucket_rows=True,
        include_all_buckets_row=False,
        save_csv=None,
    )
    chn_tbl = _fn(
        importer_iso="CHN",
        scope=scope,
        metric="value",
        years=years,
        include_individual_code_rows=False,
        include_bucket_rows=True,
        include_all_buckets_row=False,
        save_csv=None,
    )

    usa_b = usa_tbl.filter(usa_tbl["type"] == "bucket")
    chn_b = chn_tbl.filter(chn_tbl["type"] == "bucket")

    bucket_order = ["REE", "Ga", "Ge", "Ta", "B", "Co", "Ir", "P", "Si", "Ce"]

    def _extract_counts(df):
        m = {
            row["bucket"]: (int(row["k=1_cnt"]), int(row["k=2_cnt"]), int(row["k=3_cnt"]))
            for row in df.select(["bucket", "k=1_cnt", "k=2_cnt", "k=3_cnt"]).iter_rows(named=True)
        }
        vals_k1, vals_k2, vals_k3 = [], [], []
        for b in bucket_order:
            v = m.get(b, (0, 0, 0))
            vals_k1.append(v[0]); vals_k2.append(v[1]); vals_k3.append(v[2])
        return np.array(vals_k1, dtype=float), np.array(vals_k2, dtype=float), np.array(vals_k3, dtype=float)

    usa_k1, usa_k2, usa_k3 = _extract_counts(usa_b)
    chn_k1, chn_k2, chn_k3 = _extract_counts(chn_b)

    # plot format and standards
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    fig.subplots_adjust(right=0.83)

    _SCALE = 1.40
    TITLE_FS     = int(round(26 * _SCALE))
    XLABEL_FS    = int(round(26 * _SCALE))
    TICK_FS      = int(round(18 * _SCALE))
    YTICK_FS     = int(round(28 * _SCALE))
    DATALABEL_FS = int(round(22 * _SCALE))
    LEGEND_FS    = int(round(14 * _SCALE))

    y = np.arange(len(bucket_order))
    dy = 0.22
    bar_h = 0.19

    EDGE = "#0B0F19"
    LW = 1.3

    # plot USA on the right side of the plot (positive axis)
    b1 = ax.barh(y - dy,  usa_k1, height=bar_h, color=COL_USA, alpha=A1, edgecolor=EDGE, linewidth=LW, zorder=3)
    b2 = ax.barh(y,        usa_k2, height=bar_h, color=COL_USA, alpha=A2, edgecolor=EDGE, linewidth=LW, zorder=3)
    b3 = ax.barh(y + dy,   usa_k3, height=bar_h, color=COL_USA, alpha=A3, edgecolor=EDGE, linewidth=LW, zorder=3)

    # plot CHN on the left side of the plot (negative axis)
    c1 = ax.barh(y - dy, -chn_k1, height=bar_h, color=COL_CHN, alpha=A1, edgecolor=EDGE, linewidth=LW, zorder=3)
    c2 = ax.barh(y,      -chn_k2, height=bar_h, color=COL_CHN, alpha=A2, edgecolor=EDGE, linewidth=LW, zorder=3)
    c3 = ax.barh(y + dy, -chn_k3, height=bar_h, color=COL_CHN, alpha=A3, edgecolor=EDGE, linewidth=LW, zorder=3)

    # center line
    ax.axvline(0, color=EDGE, linewidth=2.0, zorder=4)

    # keep plot size symetric on both sides
    max_abs = max(
        float(np.max(np.abs(usa_k1))), float(np.max(np.abs(usa_k2))), float(np.max(np.abs(usa_k3))),
        float(np.max(np.abs(chn_k1))), float(np.max(np.abs(chn_k2))), float(np.max(np.abs(chn_k3))),
        1.0
    )
    pad = max(1.0, 0.14 * max_abs)
    ax.set_xlim(-(max_abs + pad), (max_abs + pad))

    # force the x axis label to be positive on both axes for increased understandibility and logic
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{abs(int(v))}"))
    ax.tick_params(axis="x", labelsize=TICK_FS, colors="black", width=1.4, length=7)
    ax.tick_params(axis="y", labelsize=YTICK_FS, colors="black", width=0, pad=8)

    # y labels
    ax.set_yticks(y)
    ax.set_yticklabels(bucket_order, fontsize=YTICK_FS, fontweight="black")

    # title and x axis label
    scope_short = "raw" if scope == "raw" else ("refined" if scope == "refined" else scope)
    ax.set_title(
        f"Distinct exporters in top-k set - {scope_short}",
        fontsize=TITLE_FS, fontweight="black", pad=12
    )
    ax.set_xlabel("count", fontsize=XLABEL_FS, fontweight="black", labelpad=12)

    # spines
    for s in ax.spines.values():
        s.set_linewidth(1.6)
        s.set_color("black")

    # turn gridlines off
    ax.grid(False)

    # data labels
    import matplotlib.patheffects as pe
    outline = [pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()]
    dx = 0.025 * (max_abs + pad)

    def _label_container(bar_container, align):
        for rect in bar_container:
            w = rect.get_width()
            if w == 0:
                continue
            x_end = rect.get_x() + w
            y_mid = rect.get_y() + rect.get_height() / 2
            txt = f"{int(abs(w))}"
            if align == "right":
                ax.text(
                    x_end + dx, y_mid, txt,
                    va="center", ha="left",
                    fontsize=DATALABEL_FS, fontweight="black", color="black",
                    path_effects=outline, zorder=5
                )
            else:
                ax.text(
                    x_end - dx, y_mid, txt,
                    va="center", ha="right",
                    fontsize=DATALABEL_FS, fontweight="black", color="black",
                    path_effects=outline, zorder=5
                )

    for bc in (b1, b2, b3):
        _label_container(bc, "right")
    for cc in (c1, c2, c3):
        _label_container(cc, "left")

    # legend on right side
    from matplotlib.patches import Patch

    handles_country = [
        Patch(facecolor=COL_USA, edgecolor=EDGE, linewidth=LW, label="USA"),
        Patch(facecolor=COL_CHN, edgecolor=EDGE, linewidth=LW, label="CHN"),
    ]
    spacer = Patch(facecolor="none", edgecolor="none", label="")

    # k legend uses same hue, different intensity
    handles_k = [
        Patch(facecolor=COL_USA, alpha=A1, edgecolor=EDGE, linewidth=LW, label="k=1 (dark)"),
        Patch(facecolor=COL_USA, alpha=A2, edgecolor=EDGE, linewidth=LW, label="k=2 (medium)"),
        Patch(facecolor=COL_USA, alpha=A3, edgecolor=EDGE, linewidth=LW, label="k=3 (light)"),
    ]

    handles = handles_country + [spacer] + handles_k

    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        ncols=1,
        fontsize=LEGEND_FS,
        handlelength=1.4,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.5
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_alpha(0.98)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=260, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)