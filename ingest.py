"""Raw-data ingestion, inventory, data dictionary and quality-control checks.

Raw workbooks are only ever read, never written. Two known, benign openpyxl
warnings (missing default style / unparsable header-footer in CDC exports) are
filtered by message; every other warning is left visible.
"""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import FILES, RAW, MONTHS, SUDORS_DRUGS

warnings.filterwarnings("ignore", message="Workbook contains no default style")
warnings.filterwarnings("ignore", message="Cannot parse header or footer")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs() -> None:
    """Fail early and loudly if an expected raw file is missing or unreadable."""
    for key, name in FILES.items():
        p = RAW / name
        if not p.exists():
            raise FileNotFoundError(f"Expected raw file for '{key}' not found: {p}")
        try:
            pd.ExcelFile(p)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(f"Raw file '{p.name}' could not be opened: {exc}") from exc


def load_workbooks() -> dict:
    """Return {key: {sheet_name: DataFrame}}. Values are read as-is (no coercion)."""
    out = {}
    for key, name in FILES.items():
        xl = pd.ExcelFile(RAW / name)
        out[key] = {s: xl.parse(s, header=None if "Dictionary" in s or s == "Info Tab" else 0)
                    for s in xl.sheet_names}
    return out


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------
OBS_LEVEL = {
    ("sudors", "Annual Data"): "Aggregate counts/percentages/rates: one row per jurisdiction-year (plus 'Overall' composite rows)",
    ("sudors", "Overall Trend Data"): "Aggregate: fixed-cohort composite, by year and quarter",
    ("sudors", "Trend Statistics"): "Aggregate: published first-vs-last-year change statistics",
    ("sudors", "Data Dictionary"): "Documentation",
    ("sudors", "Info Tab"): "Documentation",
    ("sys", "Jurisdiction"): "Aggregate ratio: suspected nonfatal ED overdoses per 10,000 ED visits; jurisdiction x month (and rolling-12-month annual rows)",
    ("sys", "Overall by Sex & Age"): "Aggregate ratio + percent distribution; participating jurisdictions combined; sex x age",
    ("sys", "Overall by Race & Ethnicity"): "Aggregate ratio + percent distribution; participating jurisdictions combined; race/ethnicity",
    ("sys", "Data Dictionary"): "Documentation",
    ("dx", "Jurisdiction Submission Counts"): "Aggregate: number of submitting jurisdictions by dataset-year(-month)",
    ("dx", "Jurisdiction Counts & Rates"): "Aggregate counts and crude rates per 100,000 population; jurisdiction x month/annual x ED or HOSP discharge data",
    ("dx", "County Counts & Rates"): "Aggregate counts/rates; county x year (ED discharge data only)",
    ("dx", "Overall by Sex & Age"): "Aggregate counts/rates; participating jurisdictions combined; sex x age band",
    ("dx", "Overall by Race & Ethnicity"): "Aggregate counts/rates; participating jurisdictions combined; race/ethnicity (annual)",
}

SOURCE_NOTES = {
    "sudors": ("CDC State Unintentional Drug Overdose Reporting System (SUDORS) dashboard download",
               "Fatal; data years 2020-2024; release/download date not stated inside file"),
    "sys": ("CDC Drug Overdose Surveillance and Epidemiology (DOSE) - syndromic surveillance (DOSE-SYS) dashboard download",
            "Nonfatal ED; Jan 2019-Jul 2026; file name indicates 09-28-2026"),
    "dx": ("CDC DOSE discharge-data (DOSE-DIS/'dx') dashboard download",
           "Nonfatal ED and inpatient hospitalisation; 2018-2024; file name indicates 08-05-2026"),
}


def build_inventory(wb: dict) -> pd.DataFrame:
    rows = []
    for key, name in FILES.items():
        p = RAW / name
        for sheet, df in wb[key].items():
            n_missing = int(df.isna().sum().sum())
            rows.append({
                "file": name, "dataset_key": key, "sheet": sheet,
                "file_size_bytes": p.stat().st_size, "sha256": sha256(p),
                "rows": df.shape[0], "columns": df.shape[1],
                "missing_cells(NaN)": n_missing,
                "observation_level": OBS_LEVEL.get((key, sheet), ""),
                "source": SOURCE_NOTES[key][0], "reporting_period_note": SOURCE_NOTES[key][1],
            })
    return pd.DataFrame(rows)


def build_data_dictionary(wb: dict) -> dict:
    """Return {sheet_label: DataFrame} describing every variable of every data sheet."""
    out = {}
    # SUDORS: source dictionary is inside the workbook
    d = wb["sudors"]["Data Dictionary"].copy()
    d.columns = ["variable", "description", "format", "corresponding_variable"]
    d = d.iloc[1:].reset_index(drop=True)
    out["SUDORS (source dictionary)"] = d
    # DOSE-SYS: free-text dictionary in workbook
    d2 = wb["sys"]["Data Dictionary"].copy().dropna(how="all")
    d2.columns = ["variable_or_note", "definition", "values"]
    out["DOSE-SYS (source dictionary)"] = d2.reset_index(drop=True)
    # DOSE-dx: no dictionary sheet exists -> inferred from content, flagged as such
    rows = []
    for sheet, df in wb["dx"].items():
        for c in df.columns:
            s = df[c]
            rows.append({"sheet": sheet, "variable": c, "pandas_dtype": str(s.dtype),
                         "n_missing(NaN)": int(s.isna().sum()),
                         "n_distinct": int(s.nunique(dropna=True)),
                         "example": str(s.dropna().iloc[0]) if s.notna().any() else "",
                         "note": "INFERRED from content: the DOSE-dx workbook ships without a data dictionary"})
    out["DOSE-dx (inferred)"] = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------
# Quality-control checks
# --------------------------------------------------------------------------
def qc_checks(wb: dict) -> pd.DataFrame:
    """Run explicit validation checks; every row states what was tested and the result."""
    recs = []

    def add(dataset, check, result, detail=""):
        recs.append({"dataset": dataset, "check": check, "result": result, "detail": detail})

    # ---- SUDORS ----
    a = wb["sudors"]["Annual Data"]
    ov = a["jurisdiction"].astype(str).str.startswith("Overall")
    sub = a[~ov]
    add("SUDORS", "Rows in Annual Data (jurisdiction-years + composite rows)", len(a), f"{int(ov.sum())} composite 'Overall' rows")
    add("SUDORS", "Duplicate (jurisdiction, data_year) keys", int(sub.duplicated(["jurisdiction", "data_year"]).sum()))
    cnt = sub.groupby("jurisdiction")["data_year"].nunique()
    add("SUDORS", "Jurisdictions ever included", int(cnt.size))
    add("SUDORS", "Jurisdictions included in all 5 data years (2020-2024)", int((cnt == 5).sum()),
        "Inclusion requires full reporting and >=75% circumstance information; composition changes across years")
    for d in SUDORS_DRUGS:
        cols = [f"{d}_{m}_deaths" for m in MONTHS]
        ms = sub[cols].sum(axis=1)
        ok = bool((ms == sub[f"{d}_deaths"]).all())
        add("SUDORS", f"Sum of 12 monthly counts equals annual total ({d})", "pass" if ok else "FAIL")
    n9999 = int((sub == 9999).sum().sum())
    add("SUDORS", "Cells equal to 9999 (suppressed/not available sentinel) in jurisdiction rows", n9999,
        "Treated as MISSING, never as a number; never summed")
    mc = [f"{d}_{m}_deaths" for d in SUDORS_DRUGS for m in MONTHS]
    add("SUDORS", "9999 sentinels inside monthly death counts", int((sub[mc] == 9999).sum().sum()))
    # symmetric pair variables
    eq = float((sub["imfs_heroin_deaths"] == sub["heroin_imfs_deaths"]).mean())
    add("SUDORS", "Combination variables are stored twice (imfs_heroin vs heroin_imfs identical share)", f"{eq:.3f}",
        "Unordered pairs de-duplicated before screening")
    # published cohort totals vs own aggregation
    coh = cnt[cnt == 5].index
    mine = sub[sub["jurisdiction"].isin(coh)].groupby("data_year")["alldrug_deaths"].sum()
    t = wb["sudors"]["Overall Trend Data"]
    first = t.iloc[0:5][["data_year", "alldrug_deaths"]].set_index("data_year")["alldrug_deaths"]
    same = bool((mine.sort_index().values == first.sort_index().values).all())
    add("SUDORS", "Own 27-jurisdiction cohort totals equal published 'Overall Trend Data' (2020-2024) totals",
        "pass" if same else "FAIL", f"own={mine.to_dict()} published={first.to_dict()}")
    # percent consistency: xylazine percent vs deaths/alldeaths
    x = sub[(sub["xylazine_deaths"] != 9999)]
    calc = (x["xylazine_deaths"] / x["alldrug_deaths"] * 100)
    pub_raw = x["xylazine_percent"].astype(str)
    pub = pd.to_numeric(pub_raw, errors="coerce")
    n_lt = int(pub_raw.str.startswith("<").sum())
    n_other = int(pub.isna().sum()) - n_lt
    comparable = pub.notna()
    dev = (calc[comparable] - pub[comparable]).abs()
    add("SUDORS", "Published xylazine_percent vs deaths/all-drug deaths: jurisdiction-years agreeing within 0.06 pp",
        f"{(dev <= 0.06).mean():.3f} (max abs diff {dev.max():.2f} pp)",
        f"Published 'detected' percents use a toxicology-known denominator that is not in the file; "
        f"own shares use all drug overdose deaths and are therefore conservative. "
        f"{n_lt} cell(s) published as '<0.1' excluded, {n_other} other non-numeric")

    # ---- DOSE-SYS ----
    j = wb["sys"]["Jurisdiction"]
    add("DOSE-SYS", "Rows in Jurisdiction sheet", len(j), "monthly + rolling-12-month annual rows")
    add("DOSE-SYS", "Duplicate (jurisdiction, year, month, time_unit) keys",
        int(j.duplicated(["jurisdiction", "year", "month", "time_unit"]).sum()))
    ind = [c for c in j.columns if c.endswith("_proportion")]
    for c in ind:
        s = j[(j["time_unit"] == "Monthly")][c].astype(str)
        num = pd.to_numeric(s, errors="coerce").notna().mean()
        add("DOSE-SYS", f"Share of monthly cells with a numeric value ({c})", f"{num:.3f}",
            "Non-numeric cells carry CDC symbols: * suppressed/excluded, § not shared, ¶ not participating")
    m = j[(j["time_unit"] == "Monthly")].copy()
    m["date"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1))
    gap = m[(m["date"] >= "2020-03-01") & (m["date"] <= "2020-08-01")]
    add("DOSE-SYS", "Numeric all-drug values in Mar-Aug 2020", int(pd.to_numeric(gap["drug_proportion"], errors="coerce").notna().sum()),
        "CDC excluded these months (COVID-19 onset); they are NOT zeros")

    # ---- DOSE-dx ----
    jc = wb["dx"]["Jurisdiction Counts & Rates"]
    add("DOSE-dx", "Rows in Jurisdiction Counts & Rates", len(jc))
    add("DOSE-dx", "Duplicate (jurisdiction, month, year, dataset, time_frame) keys",
        int(jc.duplicated(["jurisdiction", "month", "year", "dataset", "time_frame"]).sum()))
    st = jc[(jc["jurisdiction"] != "US") & (jc["dataset"] == "ED")].copy()
    st["cnt"] = pd.to_numeric(st["count_alldrug"], errors="coerce")
    mo = st[st["time_frame"] == "monthly"].groupby(["jurisdiction", "year"])["cnt"].sum()
    an = st[st["time_frame"] == "annual"].set_index(["jurisdiction", "year"])["cnt"]
    both = pd.concat([mo.rename("m"), an.rename("a")], axis=1).dropna()
    add("DOSE-dx", "Sum of monthly all-drug ED counts equals annual count (jurisdiction-years)",
        f"{(both['m'] == both['a']).mean():.3f}", f"n={len(both)}")
    for c in [c for c in jc.columns if c.startswith("count_")]:
        s = jc[c]
        add("DOSE-dx", f"{c}: numeric / 'suppressed' / NaN cells",
            f"{int(pd.to_numeric(s, errors='coerce').notna().sum())} / {int((s.astype(str) == 'suppressed').sum())} / {int(s.isna().sum())}",
            "NaN meaning is not documented in the source workbook; treated as unavailable, not zero")
    sub_ct = wb["dx"]["Jurisdiction Submission Counts"]
    sc = sub_ct[(sub_ct["month"].astype(str) == "all") & (sub_ct["dataset"] == "ED")].set_index("year")["jurisdiction_count"]
    mine = st[st["time_frame"] == "annual"].groupby("year")["jurisdiction"].nunique()
    add("DOSE-dx", "ED jurisdictions per year: sheet content equals 'Jurisdiction Submission Counts'",
        "pass" if (sc.sort_index().values == mine.sort_index().values).all() else "FAIL",
        f"submission={sc.to_dict()} content={mine.to_dict()}")
    return pd.DataFrame(recs)
