"""Build processed analytical datasets from the raw workbooks.

Rules applied throughout:
  * 9999 (SUDORS), '*', '§', '¶' (DOSE-SYS) and 'suppressed'/NaN (DOSE-dx) are
    converted to NaN plus an explicit status label. They are never recoded to 0.
  * Every filtering step records its before/after row counts in `FILTER_LOG`.
"""
import numpy as np
import pandas as pd

from config import (DETECTED, MONTHS, PROC, STATE_ABBR, SUDORS_DRUGS, SYS_GAP, REGION)

FILTER_LOG: list = []


def _log(step: str, before: int, after: int, note: str = "") -> None:
    FILTER_LOG.append({"step": step, "rows_before": before, "rows_after": after, "note": note})


# ---------------------------------------------------------------- SUDORS
def sudors_tables(wb: dict) -> dict:
    a = wb["sudors"]["Annual Data"].copy()
    _log("SUDORS Annual Data: all rows", len(a), len(a))
    is_ov = a["jurisdiction"].astype(str).str.startswith("Overall")
    ov = a[is_ov].copy()
    a = a[~is_ov].copy()
    _log("SUDORS: drop composite 'Overall' rows (not comparable across years)", len(a) + len(ov), len(a))
    a["abbr"] = a["jurisdiction"].map(STATE_ABBR)
    if a["abbr"].isna().any():
        raise ValueError(f"Unmapped SUDORS jurisdictions: {a.loc[a['abbr'].isna(), 'jurisdiction'].unique()}")
    a["year"] = a["data_year"].astype(int)
    ny = a.groupby("jurisdiction")["year"].transform("nunique")
    a["in_cohort27"] = ny == 5
    _log("SUDORS: jurisdictions present in all five data years (cohort of 27)",
         a["jurisdiction"].nunique(), a.loc[a["in_cohort27"], "jurisdiction"].nunique())

    # ---- monthly panel (jurisdiction x month), all jurisdiction-years
    recs = []
    for _, r in a.iterrows():
        for mi, m in enumerate(MONTHS, start=1):
            rec = {"jurisdiction": r["jurisdiction"], "abbr": r["abbr"], "year": r["year"],
                   "month": mi, "in_cohort27": r["in_cohort27"],
                   "region": REGION.get(r["abbr"], "Other")}
            for d in SUDORS_DRUGS:
                rec[d] = r[f"{d}_{m}_deaths"]
            recs.append(rec)
    mp = pd.DataFrame(recs)
    mp["date"] = pd.to_datetime(dict(year=mp["year"], month=mp["month"], day=1))
    mp["days_in_month"] = mp["date"].dt.days_in_month
    mp = mp.sort_values(["abbr", "date"]).reset_index(drop=True)

    # ---- annual categories in long form (single drugs, combinations, detected)
    cols = list(a.columns)
    i0, i1 = cols.index("opioids_stim_deaths"), cols.index("benzo_othersed_deaths")
    combo_cols = [c for c in cols[i0:i1 + 1] if c.endswith("_deaths")
                  and not c.startswith("no_") and "otherdrugs" not in c and "othdrugs" not in c]
    # drop mirror-image duplicates (e.g. imfs_heroin == heroin_imfs). Identification is by NAME
    # (unordered token pair, with 'anystim'->'stim' and 'anyopioid'->'opioids'); values are then
    # verified to be identical. Value equality alone is NOT used because columns that are entirely
    # 9999 (or entirely 0) would be merged spuriously.
    tokens = ["nonop_sed", "mdma_mda", "othopioids", "anyopioid", "rxopioid", "cathinones", "nitazenes",
              "anystim", "benzo", "cocaine", "heroin", "opioids", "stim", "meth", "imfs"]
    canon = {"anystim": "stim", "anyopioid": "opioids"}

    def pair_key(col: str):
        base = col[:-7]
        for t1 in tokens:
            if base.startswith(t1 + "_"):
                t2 = base[len(t1) + 1:]
                if t2 in tokens:
                    return frozenset({canon.get(t1, t1), canon.get(t2, t2)})
        return None

    keep, dropped, seen = [], [], {}
    for c in combo_cols:
        k = pair_key(c)
        if k is None:
            keep.append(c)
            continue
        if k in seen:
            first = seen[k]
            if a[c].equals(a[first]):
                dropped.append((c, first))
            else:  # names look mirrored but values differ: keep both, record it
                keep.append(c)
                _log("SUDORS: mirror-named columns with DIFFERENT values (both kept)", 0, 0, f"{c} vs {first}")
        else:
            seen[k] = c
            keep.append(c)
    _log("SUDORS: combination variables before/after removing mirror duplicates", len(combo_cols), len(keep),
         "; ".join(f"{b}=={a_}" for a_, b in dropped[:6]) + (" ..." if len(dropped) > 6 else ""))

    def family(c: str) -> str:
        return "exclusive/summary" if c in ("opioids_stim_deaths", "opioids_nostim_deaths", "stim_noopioids_deaths",
                                             "noopioids_nostim_deaths") or c.endswith("_only_deaths") else "combination"

    cat_rows = []
    spec = [(f"{d}_deaths", d, "single drug class", SUDORS_DRUGS[d]) for d in SUDORS_DRUGS if d != "alldrug"]
    spec += [(c, c[:-7], family(c), c[:-7]) for c in keep]
    spec += [(f"{d}_deaths", d, "detected (toxicology)", d) for d in DETECTED]
    for col, cid, fam, label in spec:
        sub = a[["jurisdiction", "abbr", "year", "in_cohort27", "alldrug_deaths", col]].copy()
        sub = sub.rename(columns={col: "deaths"})
        sub["category"] = cid
        sub["family"] = fam
        sub["label"] = label
        sub["status"] = np.where(sub["deaths"] == 9999, "not_available_9999", "value")
        sub.loc[sub["deaths"] == 9999, "deaths"] = np.nan
        cat_rows.append(sub)
    cat = pd.concat(cat_rows, ignore_index=True)

    # published annual age-adjusted rates (kept for reference only)
    rate = a[["jurisdiction", "abbr", "year", "alldrug_rate"]].copy()
    rate["alldrug_rate"] = pd.to_numeric(rate["alldrug_rate"].replace(9999, np.nan), errors="coerce")
    return {"sudors_monthly_panel": mp, "sudors_annual_categories_long": cat, "sudors_annual_rate": rate,
            "sudors_overall_rows_as_published": ov[["jurisdiction", "data_year", "alldrug_deaths", "alldrug_rate",
                                                    "opioids_deaths", "imfs_deaths", "stimulant_deaths"]],
            "sudors_dropped_mirror_columns": pd.DataFrame(dropped, columns=["dropped_column", "identical_to"])}


# ---------------------------------------------------------------- DOSE-SYS
SYS_IND = ["drug", "opioid", "fentanyl", "heroin", "stimulant", "cocaine", "methamphetamine", "benzodiazepine"]


def sys_tables(wb: dict) -> dict:
    j = wb["sys"]["Jurisdiction"].copy()
    m = j[j["time_unit"] == "Monthly"].copy()
    _log("DOSE-SYS Jurisdiction: keep monthly rows (drop rolling-12-month rows)", len(j), len(m))
    m["date"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1))
    gap0, gap1 = pd.Timestamp(SYS_GAP[0]), pd.Timestamp(SYS_GAP[1])
    out = m[["jurisdiction", "year", "month", "date"]].copy()
    for ind in SYS_IND:
        raw = m[f"{ind}_proportion"].astype(str)
        val = pd.to_numeric(raw, errors="coerce")
        status = pd.Series("value", index=m.index)
        status[raw == "*"] = "suppressed_or_excluded(*)"
        status[raw == "§"] = "not_shared(§)"
        status[raw == "¶"] = "not_participating(¶)"
        status[(raw == "*") & m["date"].between(gap0, gap1)] = "excluded_2020_gap"
        bad = val.isna() & (status == "value")
        if bad.any():
            raise ValueError(f"Unrecognised non-numeric DOSE-SYS entries in {ind}: {raw[bad].unique()}")
        out[ind] = val
        out[f"{ind}_status"] = status
    out = out.sort_values(["jurisdiction", "date"]).reset_index(drop=True)
    return {"dose_sys_monthly_clean": out}


# ---------------------------------------------------------------- DOSE-dx
DX_COUNTS = ["count_alldrug", "count_opioid", "count_fentanyl", "count_heroin", "count_stimulant",
             "count_cocaine", "count_methamphetamine", "count_benzodiazepine"]


def dx_tables(wb: dict) -> dict:
    jc = wb["dx"]["Jurisdiction Counts & Rates"].copy()
    jc = jc[jc["time_frame"] == "monthly"].copy()
    _log("DOSE-dx: keep monthly rows", len(wb["dx"]["Jurisdiction Counts & Rates"]), len(jc))
    jc["month"] = jc["month"].astype(int)
    jc["year"] = jc["year"].astype(int)
    jc["date"] = pd.to_datetime(dict(year=jc["year"], month=jc["month"], day=1))
    us = jc[jc["jurisdiction"] == "US"].copy()
    st = jc[jc["jurisdiction"] != "US"].copy()
    _log("DOSE-dx: separate 'US' composite rows from jurisdictions", len(jc), len(st))
    for c in DX_COUNTS:
        raw = st[c]
        st[f"{c}_status"] = np.where(raw.astype(str) == "suppressed", "suppressed",
                                     np.where(raw.isna(), "not_available_NaN", "value"))
        st[c] = pd.to_numeric(raw, errors="coerce")
    st["population"] = pd.to_numeric(st["population"], errors="coerce")
    st["rate_alldrug"] = pd.to_numeric(st["rate_alldrug"], errors="coerce")
    # validation: monthly crude rate per 100,000 population
    chk = (st["count_alldrug"] / st["population"] * 1e5 - st["rate_alldrug"]).abs()
    if chk.max() > 0.06:
        raise ValueError(f"DOSE-dx monthly rate does not match count/population*1e5 (max diff {chk.max():.3f})")
    keep = ["jurisdiction", "dataset", "year", "month", "date", "population"] + DX_COUNTS + \
           [f"{c}_status" for c in DX_COUNTS] + ["rate_alldrug"]
    st = st[keep].sort_values(["dataset", "jurisdiction", "date"]).reset_index(drop=True)
    return {"dose_dx_monthly_clean": st}


def build_all(wb: dict) -> dict:
    FILTER_LOG.clear()
    tabs = {}
    tabs.update(sudors_tables(wb))
    tabs.update(sys_tables(wb))
    tabs.update(dx_tables(wb))
    for name, df in tabs.items():
        df.to_csv(PROC / f"{name}.csv", index=False)
    pd.DataFrame(FILTER_LOG).to_csv(PROC / "filter_log.csv", index=False)
    return tabs
