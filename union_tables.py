import matplotlib.patheffects as pe
import polars as pl
from pathlib import Path
from typing import Literal, Iterable

def unique_exporter_table_scoped(
    importer_iso: str,
    *,
    scope: Literal["overall","raw","refined"] = "refined",
    metric: Literal["value","qty"] = "value",
    years: Iterable[int] = range(2017, 2026),
    # fallback file loading ( sed if dfs[...] isn't available)
    folder: str | Path | None = None,
    template: str | None = None, 
    #output controls
    include_individual_code_rows: bool = True,
    include_bucket_rows: bool = True,
    include_all_buckets_row: bool = True,
    save_csv: str | Path | None = None,
):
    """
    Builds a table of *unique exporters that ever appear among the top-k* suppliers
    (k=1,2,3) for the given importer, across all selected years, bucketed by your
    10 commodity families and respecting `scope` (overall/raw/refined).

    For each HS-6 code row, each bucket summary row, and an optional "All Buckets"
    row, it returns:
      - k=1_cnt, k=2_cnt, k=3_cnt  (distinct exporter count across all years)
      - k=1_iso, k=2_iso, k=3_iso  (comma-separated ISO list)

    Ranking of suppliers uses `metric = value|qty` (primaryValue vs qty).
    Data source:
      - Prefers in-memory `dfs[year]` (a Polars DataFrame with columns:
        importer, exporter, cmdCode, qty, primaryValue)
      - If not present, will attempt to load from files using `folder` + `template`.

    Notes:
      • Excludes W00 and self-trade (exporter != importer).
      • For each (bucket, year, k), the code with the *largest share loss* is chosen,
        and its top-k exporters are unioned into the cumulative set.
    """
    # -------------------- Bucket definitions --------------------
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
    def codes_for(over, raw, ref):
        return {"overall": over, "raw": raw, "refined": ref}[scope]

    #data scraping
    importer_iso = importer_iso.upper()
    metric_col = "primaryValue" if metric == "value" else "qty"

    #local cache for loaded frames, either from dfs or files
    _cache: dict[int, pl.DataFrame | None] = {}

    def _have_dfs():
        return "dfs" in globals() and isinstance(globals()["dfs"], dict)

    def _load_year(yr: int) -> pl.DataFrame | None:
        if yr in _cache:
            return _cache[yr]
        df = None
        if _have_dfs() and yr in globals()["dfs"]:
            df = globals()["dfs"][yr]
        elif folder is not None and template is not None:
            fp = Path(folder) / template.format(yr=yr, year=yr)
            if fp.exists():
                df = pl.read_csv(
                    fp, separator="\t",
                    columns=["importer","exporter","cmdCode","qty","primaryValue"],
                    infer_schema_length=1000
                )
        _cache[yr] = df
        return df

    def _filtered_frame(yr: int) -> pl.DataFrame | None:
        df = _load_year(yr)
        if df is None:
            return None
        return (
            df.filter(
                (pl.col("exporter") != "W00") &
                (pl.col("importer") != "W00") &
                (pl.col("exporter") != pl.col("importer")) &
                (pl.col("importer") == importer_iso)
            )
            .select(["importer","exporter","cmdCode","qty","primaryValue"])
        )

    #union and "top-k" logic
    k_levels = (1, 2, 3)

    def code_topk_for_year(code: int, yr: int, k: int) -> tuple[list[str], float]:
        """
        Returns (top_k_exporters_as_list, loss_percent_of_total_for_that_code_in_year)
        for the given importer, code, year, using metric_col for ranking.
        """
        base = _filtered_frame(yr)
        if base is None:
            return [], 0.0
        cdf = base.filter(pl.col("cmdCode") == code)
        if cdf.is_empty():
            return [], 0.0
        tot = float(cdf[metric_col].sum())
        if tot <= 0:
            return [], 0.0
        top = (
            cdf.group_by("exporter")
               .agg(pl.col(metric_col).sum().alias("flow"))
               .sort("flow", descending=True)
               .head(k)
        )
        expers = top["exporter"].to_list()
        loss_pct = float(top["flow"].sum()) / tot * 100.0
        return expers, loss_pct

    def union_sets_over_years(codes: set[int]) -> dict[int, set[str]]:
        """
        For a set of codes (bucket), for each k in {1,2,3}, iterate years:
          - find the code with the largest loss% (for that year & k)
          - add its top-k exporters into the set for that k.
        Returns {k: set_of_exporters}.
        """
        res = {k: set() for k in k_levels}
        if not codes:
            return res
        for yr in years:
            base = _filtered_frame(yr)
            if base is None or base.is_empty():
                continue
            for k in k_levels:
                best_loss = -1.0
                best_exp = []
                for code in codes:
                    expers, loss = code_topk_for_year(code, yr, k)
                    if loss > best_loss:
                        best_loss = loss
                        best_exp = expers
                res[k].update(best_exp)
        return res

    #building rows
    rows = []
    # per-bucket: optional per-code rows & one bucket summary row
    scoped_bucket_maps: list[tuple[str, set[int]]] = []
    for name, over, raw, ref in BUCKETS:
        scoped_codes = set(codes_for(over, raw, ref))
        scoped_bucket_maps.append((name, scoped_codes))

        if include_individual_code_rows and scoped_codes:
            for code in sorted(scoped_codes):
                s = union_sets_over_years({code})
                rows.append({
                    "label": f"{name} – {code}",
                    "bucket": name,
                    "scope": scope,
                    "type": "code",
                    **{f"k={k}_cnt": len(s[k]) for k in k_levels},
                    **{f"k={k}_iso": ", ".join(sorted(s[k])) for k in k_levels},
                })

        if include_bucket_rows and scoped_codes:
            s = union_sets_over_years(scoped_codes)
            rows.append({
                "label": f"{name} (bucket)",
                "bucket": name,
                "scope": scope,
                "type": "bucket",
                **{f"k={k}_cnt": len(s[k]) for k in k_levels},
                **{f"k={k}_iso": ", ".join(sorted(s[k])) for k in k_levels},
            })

    # optional all-buckets row (scoped union)
    if include_all_buckets_row:
        all_codes_scoped = set().union(*[codes for _, codes in scoped_bucket_maps]) if scoped_bucket_maps else set()
        s = union_sets_over_years(all_codes_scoped)
        rows.append({
            "label": "All Buckets (scoped)",
            "bucket": "ALL",
            "scope": scope,
            "type": "all_buckets",
            **{f"k={k}_cnt": len(s[k]) for k in k_levels},
            **{f"k={k}_iso": ", ".join(sorted(s[k])) for k in k_levels},
        })

    if not rows:
        print("No rows produced (check data availability, years, and scope).")
        return pl.DataFrame()

    out = pl.DataFrame(rows)

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        out.write_csv(save_csv)

    return out