"""Run the complete pipeline:  python main.py   (from 04_Python_Code/)

Stages: input validation -> inventory/QC -> harmonisation -> fatal analyses -> nonfatal analyses ->
early-warning evaluation -> robustness -> figures -> automated checks -> documentation.
"""
import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import analysis_fatal as af
import analysis_nonfatal as an
import earlywarning as ew
import figures as fg
import harmonize
import ingest
import robustness as rb
from config import DOC, FIG, N_BOOT, PROC, RAW, ROOT, SEED, SUPP

LOG = []


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def main():
    t0 = time.time()
    np.random.seed(SEED)
    log(f"Pipeline start; master seed={SEED}; bootstrap B={N_BOOT}")
    ingest.check_inputs()
    log("All three raw workbooks found and readable")
    wb = ingest.load_workbooks()

    inv = ingest.build_inventory(wb)
    inv.to_csv(DOC / "data_inventory.csv", index=False)
    dd = ingest.build_data_dictionary(wb)
    with pd.ExcelWriter(DOC / "data_dictionary.xlsx") as xw:
        for k, v in dd.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
    qc = ingest.qc_checks(wb)
    qc.to_csv(PROC / "qc_checks.csv", index=False)
    failed = qc[qc["result"].astype(str) == "FAIL"]
    if len(failed):
        raise RuntimeError(f"QC failures:\n{failed}")
    log(f"Inventory, dictionary, {len(qc)} QC checks written (no failures)")

    tabs = harmonize.build_all(wb)
    mp, cat = tabs["sudors_monthly_panel"], tabs["sudors_annual_categories_long"]
    sysm, dx = tabs["dose_sys_monthly_clean"], tabs["dose_dx_monthly_clean"]
    for d in (mp, sysm, dx):
        d["date"] = pd.to_datetime(d["date"])
    log("Harmonised tables written to 03_Processed_Data")

    cm = af.cohort_monthly(mp)
    ann, chg = af.cohort_annual(mp, cat)
    af.trend_models(cm)
    scr = af.screen_categories(cat)
    log(f"Screening: {len(scr)} eligible categories of {cat['category'].nunique()}")
    cp, det = af.changepoints_fatal(cm)
    sens, jk = af.changepoint_sensitivity(cm, mp)
    log("Fatal analyses done (trend models, screening, change points, sensitivity, jackknife)")

    an.sys_coverage(sysm)
    ser, coh, sys_ann, sys_jj = an.sys_pooled(sysm)
    an.sys_changepoints(ser)
    an.sys_state_changes(sysm)
    an.dx_pooled(dx)
    an.dx_sys_concordance(sysm, dx)
    log("Nonfatal analyses done")

    ewout = ew.run_earlywarning(mp, sysm, dx)
    log("Early-warning evaluation done (primary + 2 sensitivity configurations)")

    sc = rb.state_changes(mp)
    rb.random_effects(sc)
    rb.composition_sensitivity(mp, pd.read_csv(PROC / "sudors_overall_rows_as_published.csv"))
    rb.region_summary(sc)
    log("Geographic robustness done")

    # ---- figures
    fg.fig1_workflow()
    fg.fig2_trends(cm, ser)
    fg.fig3_shares(ann, chg)
    fg.fig4_changepoints(cm, det)
    fg.fig5_geography(sc)
    fg.fig6_earlywarning(ewout["diff"], ewout["pred"])
    fg.fig7_robustness(sens, jk)
    log("Figures 1-7 written (PNG + PDF + source-data CSV)")

    run_checks(cm, ann, chg, ewout, sens, cp)
    write_environment(t0)
    (DOC / "analysis_log.txt").write_text("\n".join(LOG) + "\n")
    log(f"Done in {time.time() - t0:.0f}s")
    (DOC / "analysis_log.txt").write_text("\n".join(LOG) + "\n")


def run_checks(cm, ann, chg, ew_out, sens, cp):
    """Automated numerical checks; any failure aborts the run."""
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"check": name, "result": "pass" if ok else "FAIL", "detail": detail})
    pub = pd.read_excel(RAW / "SUDORS-Fatal-Overdose-Data.xlsx", sheet_name="Trend Statistics")
    p0 = pub.iloc[0]
    for d, col in [("alldrug", "alldrug_deaths_relchange"), ("opioids", "opioids_deaths_relchange"), ("imfs", "imfs_deaths_relchange")]:
        mine = chg[(chg.drug == d) & (chg.from_year == 2020) & (chg.to_year == 2024) & (chg.quantity == "count")]["relative_change_pct"].iloc[0]
        chk(f"2020->2024 change in deaths ({d}) equals published Trend Statistics value", abs(mine - float(p0[col])) < 0.06,
            f"own={mine:.2f} published={p0[col]}")
    chk("Cohort monthly series sums to annual cohort totals", int(cm["alldrug"].sum()) == int(ann[ann.drug == "alldrug"]["deaths"].sum()))
    chk("Cohort has exactly 27 jurisdictions in every month", set(cm["n_jurisdictions"]) == {27})
    chk("Shares of 'alldrug' equal 100", bool((ann[ann.drug == "alldrug"]["share_pct"] == 100).all()))
    fs = pd.read_csv(FIG / "Figure4_changepoints_source_data.csv")
    obs = fs[(fs.series == "alldrug") & fs.observed.notna()]["observed"].sum()
    chk("Figure 4 observed series equals cohort monthly total", int(obs) == int(cm["alldrug"].sum()))
    chk("Early-warning evaluation uses only target months inside the declared test window",
        bool(pd.to_datetime(ew_out["pred"][ew_out["pred"].period == "test"]["T"]).between("2024-01-01", "2024-12-01").all()))
    chk("Early-warning selection table contains validation deviances only (no test period column)",
        "period" not in ew_out["sel"].columns)
    chk("Boundary flags present for all selected breakpoints", all(c in cp.columns for c in ["bp1_at_boundary"]))
    chk("Bootstrap/seed settings recorded", N_BOOT == 2000 and SEED == 20261020)
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "automated_checks.csv", index=False)
    if (out["result"] == "FAIL").any():
        raise RuntimeError(f"Automated check failure:\n{out}")
    log(f"{len(out)} automated numerical checks passed")


def write_environment(t0):
    import matplotlib, scipy, sklearn, statsmodels, ruptures
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__,
           "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "scikit-learn": sklearn.__version__,
           "matplotlib": matplotlib.__version__, "ruptures": ruptures.__version__}
    (DOC / "software_environment.txt").write_text("\n".join(f"{k}: {v}" for k, v in env.items()) + "\n")
    (DOC / "random_seeds_and_configs.txt").write_text(
        f"master seed: {SEED}\nbootstrap replicates (state-cluster): {N_BOOT}\nblock bootstrap (change points): block=6 months, B=300, seed={SEED}\n"
        f"early-warning: horizons={ew.HORIZONS}, lags={ew.LAGS}, alphas={ew.ALPHAS}, train<=2022-12, validation=2023, test=2024\n"
        f"feature sets (DOSE-SYS): {ew.FEATURE_SETS_SYS}\nminimum segment length: 6 months (sensitivity 9, 12)\n")
    (ROOT / "04_Python_Code" / "requirements.txt").write_text(
        "\n".join([f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"scipy=={scipy.__version__}",
                   f"statsmodels=={statsmodels.__version__}", f"scikit-learn=={sklearn.__version__}",
                   f"matplotlib=={matplotlib.__version__}", f"ruptures=={ruptures.__version__}", "openpyxl", "python-docx"]) + "\n")


if __name__ == "__main__":
    main()
