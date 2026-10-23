"""Nonfatal overdose indicators (DOSE-SYS syndromic, DOSE-dx discharge): descriptive trends and change points.

DOSE-SYS publishes ratios (suspected overdose ED visits per 10,000 ED visits) without visit counts,
so jurisdictions cannot be pooled by visit volume. Pooled series are therefore UNWEIGHTED means of
jurisdiction values over a fixed cohort of jurisdictions with complete numeric values (never imputed).
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis_fatal import block_bootstrap_taus, segmented_scan
from config import PROC, SEED, SYS_GAP

SYS_MAIN = ["drug", "opioid", "stimulant"]
SYS_ALL = ["drug", "opioid", "fentanyl", "heroin", "stimulant", "cocaine", "methamphetamine", "benzodiazepine"]


def sys_coverage(sysm: pd.DataFrame) -> pd.DataFrame:
    """Availability of each indicator among jurisdictions, 2021-01..2026-07 (post-gap window)."""
    w = sysm[sysm["date"] >= "2021-01-01"]
    n_months = w["date"].nunique()
    rows = []
    for ind in SYS_ALL:
        v = w.groupby("jurisdiction")[ind].apply(lambda s: s.notna().sum())
        rows.append({"indicator": ind, "months_in_window": n_months,
                     "jurisdictions_with_any_value": int((v > 0).sum()),
                     "jurisdictions_complete_all_months": int((v == n_months).sum()),
                     "share_of_jurisdiction_months_numeric": float(w[ind].notna().mean())})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "dose_sys_indicator_coverage.csv", index=False)
    return out


def sys_pooled(sysm: pd.DataFrame) -> tuple:
    w = sysm[sysm["date"] >= "2021-01-01"].copy()
    months = w["date"].nunique()
    series, cohorts = [], []
    for ind in SYS_MAIN:
        cnt = w.groupby("jurisdiction")[ind].apply(lambda s: s.notna().sum())
        coh = cnt[cnt == months].index.tolist()
        cohorts.append({"indicator": ind, "n_jurisdictions": len(coh), "jurisdictions": ",".join(sorted(coh))})
        sub = w[w["jurisdiction"].isin(coh)]
        g = sub.groupby("date")[ind].agg(["mean", "median", "count"]).reset_index()
        g["indicator"] = ind
        series.append(g)
    ser = pd.concat(series, ignore_index=True)
    ser.to_csv(PROC / "dose_sys_pooled_monthly.csv", index=False)
    pd.DataFrame(cohorts).to_csv(PROC / "dose_sys_cohorts.csv", index=False)
    # calendar-year means (2026 = Jan-Jul only) and same-month comparison Jan-Jul
    ser["year"] = ser["date"].dt.year
    ser["month"] = ser["date"].dt.month
    ann = ser.groupby(["indicator", "year"])["mean"].mean().unstack("year")
    jj = ser[ser["month"] <= 7].groupby(["indicator", "year"])["mean"].mean().unstack("year")
    ann.to_csv(PROC / "dose_sys_pooled_annual_means.csv")
    jj.to_csv(PROC / "dose_sys_pooled_janjul_means.csv")
    return ser, pd.DataFrame(cohorts), ann, jj


def sys_changepoints(ser: pd.DataFrame) -> tuple:
    rows, det = [], {}
    for ind in SYS_MAIN:
        s = ser[ser["indicator"] == ind].sort_values("date").reset_index(drop=True)
        y = np.log(s["mean"].to_numpy(float))
        mi = s["date"].dt.month.to_numpy()
        sc = segmented_scan(y, mi, max_k=2)
        k = sc["best_k"]
        rec = {"indicator": ind, "n_months": len(y), "first_month": s["date"].iloc[0].strftime("%Y-%m"),
               "last_month": s["date"].iloc[-1].strftime("%Y-%m"),
               "bic_k0": sc[0]["bic"], "bic_k1": sc[1]["bic"], "bic_k2": sc[2]["bic"], "selected_breakpoints": k}
        if k > 0:
            boots = block_bootstrap_taus(y, mi, sc[k], k)
            for i, tau in enumerate(sc[k]["taus"]):
                lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
                rec[f"bp{i+1}_month"] = s["date"].iloc[tau].strftime("%Y-%m")
                rec[f"bp{i+1}_boot95_lo"] = s["date"].iloc[int(lo)].strftime("%Y-%m")
                rec[f"bp{i+1}_boot95_hi"] = s["date"].iloc[int(hi)].strftime("%Y-%m")
                rec[f"bp{i+1}_at_boundary"] = bool(tau <= 7 or tau >= len(y) - 7)
            beta = sc[k]["beta"]
            slopes = [beta[1]] + [beta[1] + sum(beta[4:4 + j + 1]) for j in range(k)]
            for i, sl in enumerate(slopes):
                rec[f"segment{i+1}_annual_pct"] = 100 * (np.exp(sl) - 1)
        # sensitivity: drop final 3 months (possible incompleteness of newest months)
        y3, mi3 = y[:-3], mi[:-3]
        sc3 = segmented_scan(y3, mi3, max_k=2)
        rec["drop_last3_selected_k"] = sc3["best_k"]
        rec["drop_last3_breakpoints"] = ";".join(s["date"].iloc[t].strftime("%Y-%m") for t in sc3[sc3["best_k"]]["taus"]) if sc3["best_k"] else "none"
        rows.append(rec)
        det[ind] = {"s": s, "y": y, "fit": sc[k], "k": k}
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "dose_sys_changepoints.csv", index=False)
    return out, det


def sys_state_changes(sysm: pd.DataFrame) -> pd.DataFrame:
    """Jurisdiction-level change in the SYS all-drug indicator: mean of 2025 months vs mean of 2023 months."""
    w = sysm[sysm["year"].isin([2023, 2025])]
    rows = []
    for ind in SYS_MAIN:
        g = w.groupby(["jurisdiction", "year"])[ind].agg(["mean", "count"]).unstack("year")
        ok = g[(g[("count", 2023)] == 12) & (g[("count", 2025)] == 12)]
        ratio = ok[("mean", 2025)] / ok[("mean", 2023)]
        rows.append({"indicator": ind, "n_jurisdictions": len(ok), "median_ratio_2025_vs_2023": float(ratio.median()),
                     "share_declined": float((ratio < 1).mean()),
                     "iqr_lo": float(ratio.quantile(0.25)), "iqr_hi": float(ratio.quantile(0.75))})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "dose_sys_state_changes.csv", index=False)
    return out


# ------------------------------------------------------------------ DOSE-dx (discharge)
def dx_pooled(dx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in ["ED", "HOSP"]:
        d = dx[dx["dataset"] == ds]
        for ind in ["count_alldrug", "count_opioid", "count_stimulant"]:
            st = f"{ind}_status"
            ok = d.groupby("jurisdiction")[st].apply(lambda s: (s == "value").all())
            coh = ok[ok].index
            sub = d[d["jurisdiction"].isin(coh)]
            for y, g in sub.groupby("year"):
                pop_year = g.groupby("jurisdiction")["population"].first().sum()
                tot = g[ind].sum()
                rows.append({"dataset": ds, "indicator": ind, "year": int(y), "n_jurisdictions": len(coh),
                             "count": int(tot), "population": int(pop_year), "crude_rate_per_100k": 1e5 * tot / pop_year})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "dose_dx_pooled_annual.csv", index=False)
    return out


def dx_sys_concordance(sysm: pd.DataFrame, dx: pd.DataFrame) -> pd.DataFrame:
    """Within-jurisdiction monthly Spearman correlation between SYS all-drug proportion and DOSE-dx ED all-drug rate.

    The two quantities have different denominators (per 10,000 ED visits vs per 100,000 population) and
    different case definitions; only rank concordance is assessed.
    """
    e = dx[dx["dataset"] == "ED"][["jurisdiction", "date", "rate_alldrug"]]
    m = sysm[["jurisdiction", "date", "drug"]].merge(e, on=["jurisdiction", "date"]).dropna()
    rows = []
    for j, g in m.groupby("jurisdiction"):
        if len(g) >= 24:
            r, p = stats.spearmanr(g["drug"], g["rate_alldrug"])
            rows.append({"jurisdiction": j, "n_months": len(g), "spearman_rho": r, "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "dose_sys_dx_concordance.csv", index=False)
    return out
