"""Fatal-overdose (SUDORS) analyses: RQ1 (temporal change), RQ2 (emerging patterns, change points).

All inference is conditional on the SUDORS jurisdiction-inclusion rules. The primary
analytic population is the fixed cohort of 27 jurisdictions that met inclusion criteria
in every data year 2020-2024, because the dashboard's 'Overall' composites change
membership from year to year and are not comparable across years.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson

from config import N_BOOT, PROC, SEED, SUDORS_DRUGS

YEARS = [2020, 2021, 2022, 2023, 2024]


# ------------------------------------------------------------------ helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def cohort_monthly(mp: pd.DataFrame) -> pd.DataFrame:
    c = mp[mp["in_cohort27"]]
    g = c.groupby("date")[list(SUDORS_DRUGS)].sum()
    g["days_in_month"] = g.index.days_in_month
    g["n_jurisdictions"] = c.groupby("date")["abbr"].nunique()
    g = g.reset_index()
    g.to_csv(PROC / "fatal_cohort27_monthly.csv", index=False)
    return g


def cohort_annual(mp: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Annual cohort totals, share of all-drug deaths and changes (state-cluster bootstrap CIs)."""
    rng = np.random.default_rng(SEED)
    c = mp[mp["in_cohort27"]]
    by_state = c.groupby(["abbr", "year"])[list(SUDORS_DRUGS)].sum()
    states = by_state.index.get_level_values(0).unique().to_numpy()
    rows = []
    for d in SUDORS_DRUGS:
        tab = by_state[d].unstack("year")
        tot = by_state["alldrug"].unstack("year")
        for y in YEARS:
            k, n = tab[y].sum(), tot[y].sum()
            lo, hi = wilson(k, n)
            rows.append({"drug": d, "year": y, "deaths": int(k), "all_drug_deaths": int(n),
                         "share_pct": 100 * k / n, "share_lo_wilson": 100 * lo, "share_hi_wilson": 100 * hi})
    ann = pd.DataFrame(rows)

    # bootstrap (resample jurisdictions with replacement) for relative changes in counts and shares
    arr = {d: by_state[d].unstack("year").loc[states, YEARS].to_numpy(float) for d in SUDORS_DRUGS}
    yi = {y: i for i, y in enumerate(YEARS)}
    boot_idx = [rng.integers(0, len(states), size=len(states)) for _ in range(N_BOOT)]

    def ratio(idx, d, y0, y1, kind):
        num, den0 = arr[d][idx], arr["alldrug"][idx]
        if kind == "count":
            return num[:, yi[y1]].sum() / num[:, yi[y0]].sum() - 1
        return (num[:, yi[y1]].sum() / den0[:, yi[y1]].sum()) / (num[:, yi[y0]].sum() / den0[:, yi[y0]].sum()) - 1

    chg = []
    full = np.arange(len(states))
    for d in SUDORS_DRUGS:
        for (y0, y1) in [(2020, 2024), (2023, 2024), (2020, 2023)]:
            for kind in ["count", "share"]:
                if d == "alldrug" and kind == "share":
                    continue
                est = ratio(full, d, y0, y1, kind)
                boots = [ratio(ix, d, y0, y1, kind) for ix in boot_idx]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                chg.append({"drug": d, "from_year": y0, "to_year": y1, "quantity": kind,
                            "relative_change_pct": 100 * est, "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
                            "ci_method": f"percentile bootstrap resampling {len(states)} jurisdictions, B={N_BOOT}"})
    chg = pd.DataFrame(chg)
    ann.to_csv(PROC / "fatal_cohort27_annual.csv", index=False)
    chg.to_csv(PROC / "fatal_cohort27_changes.csv", index=False)
    return ann, chg


# ------------------------------------------------------------------ trend models
def _design(n, month_idx, extra=None):
    t = np.arange(n) / 12.0
    X = pd.DataFrame({"const": 1.0, "t_years": t,
                      "sin12": np.sin(2 * np.pi * month_idx / 12), "cos12": np.cos(2 * np.pi * month_idx / 12)})
    if extra is not None:
        for k, v in extra.items():
            X[k] = v
    return X


def trend_models(cm: pd.DataFrame) -> pd.DataFrame:
    """Quasi-Poisson (HAC) log-linear trend with offset log(days); NB as sensitivity; diagnostics."""
    rows = []
    for d in SUDORS_DRUGS:
        for label, sel in [("2020-01..2024-12", slice(None)), ("2020-01..2023-12", slice(0, 48))]:
            s = cm.iloc[sel]
            y = s[d].to_numpy(float)
            X = _design(len(s), s["date"].dt.month.to_numpy())
            off = np.log(s["days_in_month"].to_numpy(float))
            m = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            b, ci = m.params["t_years"], m.conf_int().loc["t_years"]
            pear = float((m.resid_pearson ** 2).sum() / m.df_resid)
            lb = acorr_ljungbox(m.resid_response / np.sqrt(m.fittedvalues), lags=[12], return_df=True)["lb_pvalue"].iloc[0]
            dw = durbin_watson(m.resid_pearson)
            # NB sensitivity (alpha by moment estimator)
            alpha = max(1e-6, float(((y - m.fittedvalues) ** 2 - m.fittedvalues).sum() / (m.fittedvalues ** 2).sum()))
            nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), offset=off).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            rows.append({"drug": d, "window": label, "n_months": len(s),
                         "annual_pct_change": 100 * (np.exp(b) - 1),
                         "ci_lo_pct": 100 * (np.exp(ci[0]) - 1), "ci_hi_pct": 100 * (np.exp(ci[1]) - 1),
                         "p_value": float(m.pvalues["t_years"]),
                         "pearson_dispersion": pear, "ljung_box_p_lag12": float(lb), "durbin_watson": float(dw),
                         "nb_annual_pct_change": 100 * (np.exp(nb.params["t_years"]) - 1),
                         "nb_alpha": alpha})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "fatal_trend_models.csv", index=False)
    return out


# ------------------------------------------------------------------ screening
def screen_categories(cat: pd.DataFrame, min_jur: int = 10, min_total_2024: int = 100) -> pd.DataFrame:
    """Test annual change in the SHARE of all drug overdose deaths involving each category.

    * Population: jurisdictions with a valid (non-9999) count in all five years for that category.
    * Model: binomial GLM, logit(share) ~ year (centred), cluster-robust SE by jurisdiction.
    * Two exploratory slope tests per category: full period (2020-2024) and recent (2022-2024).
    * Multiple comparisons: Benjamini-Hochberg across all eligible categories, separately per window.
    """
    rows = []
    for cid, g in cat.groupby("category"):
        piv = g.pivot_table(index="abbr", columns="year", values="deaths", aggfunc="first")
        den = g.pivot_table(index="abbr", columns="year", values="alldrug_deaths", aggfunc="first")
        ok = piv.dropna().index
        if len(ok) < min_jur:
            continue
        piv, den = piv.loc[ok], den.loc[ok]
        fam = g["family"].iloc[0]
        total24 = piv[2024].sum()
        if total24 < min_total_2024:
            continue
        rec = {"category": cid, "family": fam, "n_jurisdictions": len(ok),
               "needs_note": ""}
        for y in YEARS:
            rec[f"deaths_{y}"] = int(piv[y].sum())
            rec[f"share_{y}_pct"] = 100 * piv[y].sum() / den[y].sum()
        rec["count_change_2020_2024_pct"] = 100 * (piv[2024].sum() / piv[2020].sum() - 1) if piv[2020].sum() > 0 else np.nan
        rec["count_change_2023_2024_pct"] = 100 * (piv[2024].sum() / piv[2023].sum() - 1) if piv[2023].sum() > 0 else np.nan
        rec["share_change_pp_2020_2024"] = rec["share_2024_pct"] - rec["share_2020_pct"]
        for win, yrs in [("full", YEARS), ("recent", [2022, 2023, 2024])]:
            long = []
            for y in yrs:
                for s in ok:
                    long.append((s, y - 2022, piv.loc[s, y], den.loc[s, y] - piv.loc[s, y]))
            L = pd.DataFrame(long, columns=["abbr", "yc", "succ", "fail"])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # convergence/separation problems become failures, not silent results
                    m = sm.GLM(L[["succ", "fail"]].to_numpy(), sm.add_constant(L[["yc"]]), family=sm.families.Binomial()).fit(
                        cov_type="cluster", cov_kwds={"groups": pd.factorize(L["abbr"])[0], "use_correction": True})
                b = m.params["yc"]
                ci = m.conf_int().loc["yc"]
                rec[f"{win}_or_per_year"] = float(np.exp(b))
                rec[f"{win}_or_lo"], rec[f"{win}_or_hi"] = float(np.exp(ci[0])), float(np.exp(ci[1]))
                rec[f"{win}_p"] = float(m.pvalues["yc"])
            except Exception as exc:  # noqa: BLE001
                rec[f"{win}_p"] = np.nan
                rec["needs_note"] += f"{win}: model failed ({type(exc).__name__}); "
        rows.append(rec)
    out = pd.DataFrame(rows)
    for win in ["full", "recent"]:
        ok = out[f"{win}_p"].notna()
        q = np.full(len(out), np.nan)
        q[ok.to_numpy()] = multipletests(out.loc[ok, f"{win}_p"], method="fdr_bh")[1]
        out[f"{win}_q_bh"] = q
    # exploratory descriptive label (NOT a claim of novelty or causal emergence)
    def lab(r):
        if pd.notna(r["recent_q_bh"]) and r["recent_q_bh"] < 0.05 and r["recent_or_per_year"] > 1:
            return "share rising 2022-24 (count up)" if r["count_change_2023_2024_pct"] > 0 else "share rising 2022-24 (count down)"
        if pd.notna(r["recent_q_bh"]) and r["recent_q_bh"] < 0.05:
            return "share falling 2022-24"
        return "no credible share change 2022-24"
    out["exploratory_label"] = out.apply(lab, axis=1)
    out = out.sort_values("recent_q_bh")
    out.to_csv(PROC / "fatal_category_screening.csv", index=False)
    return out


# ------------------------------------------------------------------ change points
def _hinge_design(n, month_idx, taus):
    t = np.arange(n)
    cols = {"const": np.ones(n), "t": t / 12.0,
            "sin12": np.sin(2 * np.pi * month_idx / 12), "cos12": np.cos(2 * np.pi * month_idx / 12)}
    for i, tau in enumerate(taus):
        cols[f"h{i}"] = np.maximum(0, t - tau) / 12.0
    return np.column_stack(list(cols.values()))


def _rss(y, X):
    beta, res, rk, _ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r), beta, r


def segmented_scan(y, month_idx, min_seg=6, max_k=2):
    """Continuous piecewise-linear (joinpoint-style) model on log scale, exhaustive grid search.

    y: log deaths per day (or other log-scale series). Returns dict with best model per K and BIC choice.
    BIC counts each breakpoint as one extra parameter.
    """
    n = len(y)
    res = {}
    rss0, b0, r0 = _rss(y, _hinge_design(n, month_idx, []))
    res[0] = {"taus": [], "rss": rss0, "k_params": 4, "beta": b0, "resid": r0}
    grid = list(range(min_seg, n - min_seg + 1))
    if max_k >= 1:
        best = None
        prof = {}
        for tau in grid:
            rss, b, r = _rss(y, _hinge_design(n, month_idx, [tau]))
            prof[tau] = rss
            if best is None or rss < best["rss"]:
                best = {"taus": [tau], "rss": rss, "k_params": 6, "beta": b, "resid": r}
        best["profile"] = prof
        res[1] = best
    if max_k >= 2:
        best = None
        for i, t1 in enumerate(grid):
            for t2 in grid[i + 1:]:
                if t2 - t1 < min_seg:
                    continue
                rss, b, r = _rss(y, _hinge_design(n, month_idx, [t1, t2]))
                if best is None or rss < best["rss"]:
                    best = {"taus": [t1, t2], "rss": rss, "k_params": 8, "beta": b, "resid": r}
        res[2] = best
    for k, m in res.items():
        m["bic"] = n * np.log(m["rss"] / n) + m["k_params"] * np.log(n)
    res["best_k"] = min([k for k in res if isinstance(k, int)], key=lambda k: res[k]["bic"])
    return res


def block_bootstrap_taus(y, month_idx, fit, k, block=6, B=300, min_seg=6, seed=SEED):
    """Moving-block residual bootstrap of the breakpoint location(s) under the selected K model."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fitted = y - fit["resid"]
    out = []
    nb = int(np.ceil(n / block))
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, size=nb)
        e = np.concatenate([fit["resid"][s:s + block] for s in starts])[:n]
        ys = fitted + e
        r = segmented_scan(ys, month_idx, min_seg=min_seg, max_k=k)
        out.append(r[k]["taus"] if k in r else [])
    return np.array(out)


def profile_ci(prof, rss_min, n, crit=3.84):
    """Likelihood-ratio style set {tau: n*log(RSS/RSS_min) <= crit} for K=1 (approximate)."""
    ok = [t for t, r in prof.items() if n * np.log(r / rss_min) <= crit]
    return (min(ok), max(ok)) if ok else (np.nan, np.nan)


def changepoints_fatal(cm: pd.DataFrame) -> tuple:
    import ruptures as rpt
    rows, details = [], {}
    for d in SUDORS_DRUGS:
        y = np.log(cm[d].to_numpy(float) / cm["days_in_month"].to_numpy(float))
        mi = cm["date"].dt.month.to_numpy()
        scan = segmented_scan(y, mi)
        k = scan["best_k"]
        fit = scan[k]
        boots = block_bootstrap_taus(y, mi, fit, k) if k > 0 else np.empty((0, 0))
        dates = cm["date"]
        rec = {"series": d, "n_months": len(y), "bic_k0": scan[0]["bic"], "bic_k1": scan[1]["bic"], "bic_k2": scan[2]["bic"],
               "selected_breakpoints": k}
        if k > 0:
            for i, tau in enumerate(fit["taus"]):
                rec[f"bp{i+1}_month"] = dates.iloc[tau].strftime("%Y-%m")
                # a break within one month of the admissible limit is a boundary solution, not an interior change point
                rec[f"bp{i+1}_at_boundary"] = bool(tau <= 6 + 1 or tau >= len(y) - 6 - 1)
                if len(boots):
                    lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
                    rec[f"bp{i+1}_boot95_lo"] = dates.iloc[int(lo)].strftime("%Y-%m")
                    rec[f"bp{i+1}_boot95_hi"] = dates.iloc[int(hi)].strftime("%Y-%m")
            if k == 1:
                lo, hi = profile_ci(scan[1]["profile"], scan[1]["rss"], len(y))
                rec["bp1_profile_lo"] = dates.iloc[int(lo)].strftime("%Y-%m")
                rec["bp1_profile_hi"] = dates.iloc[int(hi)].strftime("%Y-%m")
            # slopes per segment (annual % change, on log scale)
            beta = fit["beta"]
            slopes = [beta[1]] + [beta[1] + sum(beta[4:4 + j + 1]) for j in range(k)]
            for i, sl in enumerate(slopes):
                rec[f"segment{i+1}_annual_pct"] = 100 * (np.exp(sl) - 1)
        # PELT / BinSeg on seasonally-adjusted log series (level-shift detectors; alternative family)
        Xs = _hinge_design(len(y), mi, [])
        _, bs, _ = _rss(y, Xs)
        adj = y - Xs[:, 2:4] @ bs[2:4]
        sd = np.std(np.diff(adj)) / np.sqrt(2)
        for scale in (1, 3):
            algo = rpt.Pelt(model="l2", min_size=6, jump=1).fit(adj.reshape(-1, 1))
            bk = algo.predict(pen=scale * 2 * np.log(len(y)) * sd ** 2)[:-1]
            rec[f"pelt_pen{scale}x"] = ";".join(dates.iloc[b].strftime("%Y-%m") for b in bk) or "none"
        bs_ = rpt.Binseg(model="l2", min_size=6).fit(adj.reshape(-1, 1)).predict(n_bkps=1)[:-1]
        rec["binseg_1bp"] = ";".join(dates.iloc[b].strftime("%Y-%m") for b in bs_)
        rows.append(rec)
        details[d] = {"y": y, "fit": fit, "k": k, "boots": boots}
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "fatal_changepoints.csv", index=False)
    return out, details


def changepoint_sensitivity(cm: pd.DataFrame, mp: pd.DataFrame) -> pd.DataFrame:
    """Vary specification for the K=1 breakpoint of key series; add jackknife over jurisdictions."""
    rows = []
    key = ["alldrug", "opioids", "imfs", "stimulant", "meth", "cocaine"]
    for d in key:
        def run(label, series_df, min_seg=6, trim_end=0, glm=False):
            s = series_df.iloc[: len(series_df) - trim_end] if trim_end else series_df
            y = np.log(s[d].to_numpy(float) / s["days_in_month"].to_numpy(float))
            mi = s["date"].dt.month.to_numpy()
            sc = segmented_scan(y, mi, min_seg=min_seg, max_k=1)
            tau = sc[1]["taus"][0]
            rows.append({"series": d, "specification": label, "bp_month": s["date"].iloc[tau].strftime("%Y-%m"),
                         "k_selected_by_bic": sc["best_k"]})
        run("baseline (min segment 6)", cm)
        run("min segment 9", cm, min_seg=9)
        run("min segment 12", cm, min_seg=12)
        run("drop final 3 months", cm, trim_end=3)
        run("drop final 6 months", cm, trim_end=6)
        run("series starts 2021-01 (excludes 2020)", cm.iloc[12:].reset_index(drop=True))
        # quasi-Poisson segmented (deviance profile) for K=1
        y = cm[d].to_numpy(float)
        n = len(y)
        mi = cm["date"].dt.month.to_numpy()
        off = np.log(cm["days_in_month"].to_numpy(float))
        best = None
        for tau in range(6, n - 5):
            X = _hinge_design(n, mi, [tau])
            m = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
            if best is None or m.deviance < best[1]:
                best = (tau, m.deviance)
        rows.append({"series": d, "specification": "Poisson deviance profile (K=1)",
                     "bp_month": cm["date"].iloc[best[0]].strftime("%Y-%m"), "k_selected_by_bic": np.nan})
    out = pd.DataFrame(rows)
    # jackknife: leave one jurisdiction out (K=1 location, log-linear segmented)
    jk = []
    c = mp[mp["in_cohort27"]]
    for d in ["alldrug", "imfs", "stimulant"]:
        for s in sorted(c["abbr"].unique()):
            g = c[c["abbr"] != s].groupby("date")[[d]].sum().reset_index()
            g["days_in_month"] = g["date"].dt.days_in_month
            y = np.log(g[d].to_numpy(float) / g["days_in_month"].to_numpy(float))
            sc = segmented_scan(y, g["date"].dt.month.to_numpy(), max_k=1)
            jk.append({"series": d, "left_out": s, "bp_month": g["date"].iloc[sc[1]["taus"][0]].strftime("%Y-%m")})
    jk = pd.DataFrame(jk)
    out.to_csv(PROC / "fatal_changepoint_sensitivity.csv", index=False)
    jk.to_csv(PROC / "fatal_changepoint_jackknife.csv", index=False)
    return out, jk
