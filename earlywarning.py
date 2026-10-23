"""RQ3 - do nonfatal overdose indicators add out-of-sample information about later fatal overdose counts?

Design (all choices fixed before the test period was evaluated):
  * Unit: jurisdiction-month.  Target: SUDORS all-drug overdose deaths in month T = t + h (h = 1, 2, 3).
  * Origin t: month at which the nonfatal indicator (DOSE-SYS or DOSE-dx) is known.
  * Reporting-lag scenario L (0, 3, 6): fatal history is only observed through month t - L.
  * Chronological split by TARGET month: training <= 2022-12, validation 2023, test 2024.
  * Rolling-origin evaluation: for each evaluation target month the model is re-fitted on all rows whose
    targets were already observed at the forecast origin (target month <= t - L). No future rows are used.
  * Model selection (feature set, ridge penalty) uses the VALIDATION period only; the test period is
    evaluated once with the frozen configuration.
  * Baselines: persistence, 3-month moving average, seasonal naive, and a Poisson GLM using only fatal
    history (the 'nonfatal-free' model on exactly the same rows). The augmented model adds nonfatal
    momentum features to that GLM.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor

from config import N_BOOT, PROC, SEED

SPLIT = {"validation": ("2023-01-01", "2023-12-01"), "test": ("2024-01-01", "2024-12-01")}
TRAIN_END = pd.Timestamp("2022-12-01")
HORIZONS = [1, 2, 3]
LAGS = [0, 3, 6]
ALPHAS = [0.0, 0.01, 0.1, 1.0]
FEATURE_SETS_SYS = [("drug",), ("opioid",), ("stimulant",), ("drug", "opioid"), ("drug", "stimulant"),
                    ("opioid", "stimulant"), ("drug", "opioid", "stimulant")]


# ------------------------------------------------------------------ panel construction
def _wide(df, value, col="abbr"):
    w = df.pivot_table(index="date", columns=col, values=value, aggfunc="first")
    idx = pd.date_range("2019-01-01", "2026-07-01", freq="MS")
    return w.reindex(idx)


def build_wide(mp, sysm, dx, scope="cohort27"):
    f = mp if scope == "all_available" else mp[mp["in_cohort27"]]
    F = _wide(f, "alldrug")
    S = {k: _wide(sysm.rename(columns={"jurisdiction": "abbr"}), k) for k in ["drug", "opioid", "stimulant"]}
    e = dx[dx["dataset"] == "ED"].rename(columns={"jurisdiction": "abbr"})
    D = {"ed_alldrug": _wide(e, "rate_alldrug")}
    return F, S, D


def _momentum(W):
    """log(mean of last 3 months) - log(mean of the 3 months before). Zero means have no defined log-ratio and
    are set to NaN (row excluded), never to an arbitrary constant."""
    r = W.rolling(3, min_periods=3).mean()
    r = r.where(r > 0)
    return np.log(r) - np.log(r.shift(3))


def make_rows(F, feats, L, h):
    """Rows indexed by (origin month t, jurisdiction) with target at t+h and fatal history ending at t-L."""
    y = F.shift(-h)
    y_last = F.shift(L)
    ma3 = F.rolling(3, min_periods=3).mean().shift(L)
    snaive = F.shift(12 - h)
    cols = {"y": y, "y_last": y_last, "ma3": ma3, "snaive": snaive}
    out = {k: v.stack(future_stack=True) for k, v in cols.items()}
    for name, W in feats.items():
        out[f"z_{name}"] = _momentum(W).reindex(F.index).stack(future_stack=True)
    R = pd.DataFrame(out)
    R.index.names = ["t", "abbr"]
    R = R.reset_index()
    R["T"] = R["t"] + pd.DateOffset(months=h)
    R["x_ar"] = np.log((R["y_last"] + 0.5) / (R["ma3"] + 0.5))
    R["expo"] = R["ma3"] + 0.5
    return R


def deviance(y, mu):
    mu = np.maximum(mu, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def _fit_predict(train, test, cols, alpha):
    mu_x, sd_x = train[cols].mean(), train[cols].std(ddof=0).replace(0, 1)
    Xtr = ((train[cols] - mu_x) / sd_x).to_numpy()
    Xte = ((test[cols] - mu_x) / sd_x).to_numpy()
    r = (train["y"] / train["expo"]).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a non-converged fit must fail loudly
        m = PoissonRegressor(alpha=alpha, max_iter=2000, solver="newton-cholesky" if alpha == 0 else "lbfgs")
        m.fit(Xtr, r, sample_weight=train["expo"].to_numpy())
    mu_tr = train["expo"].to_numpy() * m.predict(Xtr)
    mu_te = test["expo"].to_numpy() * m.predict(Xte)
    disp = max(0.0, float((((train["y"] - mu_tr) ** 2 - mu_tr).sum()) / (mu_tr ** 2).sum()))
    return mu_te, disp


def evaluate(F, feats, feature_cols, period, L, h, alpha_base, alpha_aug, min_train_rows=60):
    """Rolling-origin predictions for one (period, L, h). feature_cols = list of z_* columns for augmented model."""
    R = make_rows(F, feats, L, h)
    need = ["y", "y_last", "ma3", "snaive", "x_ar"] + feature_cols
    R = R.dropna(subset=need)
    # identical rows for every model: require all candidate z features where available is handled by caller
    lo, hi = pd.Timestamp(SPLIT[period][0]), pd.Timestamp(SPLIT[period][1])
    rows = []
    for T in pd.date_range(lo, hi, freq="MS"):
        t = T - pd.DateOffset(months=h)
        train = R[R["T"] <= t - pd.DateOffset(months=L)]
        test = R[R["T"] == T]
        if len(train) < min_train_rows or test.empty:
            continue
        mu_b, d_b = _fit_predict(train, test, ["x_ar"], alpha_base)
        mu_a, d_a = _fit_predict(train, test, ["x_ar"] + feature_cols, alpha_aug)
        te = test.copy()
        te["mu_glm_base"], te["disp_glm_base"] = mu_b, d_b
        te["mu_glm_aug"], te["disp_glm_aug"] = mu_a, d_a
        te["mu_persistence"] = te["y_last"]
        te["mu_ma3"] = te["ma3"]
        te["mu_snaive"] = te["snaive"]
        te["n_train"] = len(train)
        rows.append(te)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["period"], out["L"], out["h"] = period, L, h
    return out


MODELS = ["persistence", "ma3", "snaive", "glm_base", "glm_aug"]


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, L, h), g in pred.groupby(["period", "L", "h"]):
        for m in MODELS:
            mu = np.maximum(g[f"mu_{m}"].to_numpy(float), 0.5)
            y = g["y"].to_numpy(float)
            rec = {"period": period, "L": L, "h": h, "model": m, "n_rows": len(g), "n_jurisdictions": g["abbr"].nunique(),
                   "mae": np.mean(np.abs(y - mu)), "rmse": np.sqrt(np.mean((y - mu) ** 2)),
                   "mean_poisson_deviance": deviance(y, mu).mean()}
            if m in ("glm_base", "glm_aug"):
                a = g[f"disp_{m}"].to_numpy(float)
                cov80 = []
                cov95 = []
                for q_lo, q_hi, store in [(0.1, 0.9, cov80), (0.025, 0.975, cov95)]:
                    n = np.where(a > 1e-8, 1.0 / np.maximum(a, 1e-8), 1e8)
                    p = n / (n + mu)
                    lo = stats.nbinom.ppf(q_lo, n, p)
                    hi = stats.nbinom.ppf(q_hi, n, p)
                    store.append(np.mean((y >= lo) & (y <= hi)))
                rec["coverage80"], rec["coverage95"] = cov80[0], cov95[0]
            rows.append(rec)
    out = pd.DataFrame(rows)
    # aggregate (sum over jurisdictions) absolute percentage error by target month
    ag = []
    for (period, L, h), g in pred.groupby(["period", "L", "h"]):
        for m in MODELS:
            s = g.groupby("T").agg(y=("y", "sum"), mu=(f"mu_{m}", "sum"))
            ag.append({"period": period, "L": L, "h": h, "model": m,
                       "aggregate_mape_pct": float((np.abs(s["y"] - s["mu"]) / s["y"]).mean() * 100),
                       "aggregate_bias_pct": float(((s["mu"] - s["y"]).sum() / s["y"].sum()) * 100)})
    return out.merge(pd.DataFrame(ag), on=["period", "L", "h", "model"])


def cluster_bootstrap_diff(pred: pd.DataFrame, m_a: str, m_b: str, B=N_BOOT, seed=SEED):
    """Mean deviance(m_a) - mean deviance(m_b); negative favours m_a. Resamples jurisdictions."""
    out = []
    rng = np.random.default_rng(seed)
    for (period, L, h), g in pred.groupby(["period", "L", "h"]):
        da = deviance(g["y"].to_numpy(float), g[f"mu_{m_a}"].to_numpy(float))
        db = deviance(g["y"].to_numpy(float), g[f"mu_{m_b}"].to_numpy(float))
        codes, uniq = pd.factorize(g["abbr"])
        k = len(uniq)
        sa, sb = np.bincount(codes, da, k), np.bincount(codes, db, k)
        cnt = np.bincount(codes, minlength=k)
        est = (sa.sum() - sb.sum()) / cnt.sum()
        boots = []
        for _ in range(B):
            idx = rng.integers(0, k, k)
            boots.append((sa[idx].sum() - sb[idx].sum()) / cnt[idx].sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        per_state = (sa - sb) / cnt
        out.append({"period": period, "L": L, "h": h, "model_a": m_a, "model_b": m_b,
                    "mean_deviance_diff": est, "ci_lo": lo, "ci_hi": hi,
                    "base_mean_deviance": sb.sum() / cnt.sum(),
                    "relative_diff_pct": 100 * est / (sb.sum() / cnt.sum()),
                    "rel_ci_lo_pct": 100 * lo / (sb.sum() / cnt.sum()), "rel_ci_hi_pct": 100 * hi / (sb.sum() / cnt.sum()),
                    "n_jurisdictions_a_better": int((per_state < 0).sum()), "n_jurisdictions": k})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ selection and full run
def select_config(F, feats_all, feature_sets, name_map):
    """Choose (feature set, alpha) for the augmented model and alpha for the baseline on VALIDATION only (L=0).

    All candidate configurations see an identical row set (every candidate z feature available)."""
    zc_all = [f"z_{k}" for k in name_map]
    feats_sub = {k: feats_all[k] for k in name_map}
    Rs = {h: make_rows(F, feats_sub, 0, h).dropna(subset=["y", "y_last", "ma3", "snaive", "x_ar"] + zc_all) for h in HORIZONS}
    res = []
    for fs in feature_sets:
        cols = [f"z_{k}" for k in fs]
        for a in ALPHAS:
            devs = []
            for h in HORIZONS:
                pred = _eval_on_rows(Rs[h], h, "validation", cols, a)
                if not pred.empty:
                    devs.append(deviance(pred["y"].to_numpy(float), pred["mu_glm_aug"].to_numpy(float)).mean())
            res.append({"feature_set": "+".join(fs), "alpha": a,
                        "validation_mean_deviance": float(np.mean(devs)) if devs else np.nan, "n_horizons": len(devs)})
    sel = pd.DataFrame(res)
    base = []
    for a in ALPHAS:
        devs = []
        for h in HORIZONS:
            pred = _eval_on_rows(Rs[h], h, "validation", [zc_all[0]], a, base_only=True)
            if not pred.empty:
                devs.append(deviance(pred["y"].to_numpy(float), pred["mu_glm_base"].to_numpy(float)).mean())
        base.append({"alpha": a, "validation_mean_deviance": float(np.mean(devs)) if devs else np.nan})
    return sel, pd.DataFrame(base)


def _eval_on_rows(R, h, period, feature_cols, alpha, L=0, base_only=False, alpha_base=None):
    lo, hi = pd.Timestamp(SPLIT[period][0]), pd.Timestamp(SPLIT[period][1])
    rows = []
    for T in pd.date_range(lo, hi, freq="MS"):
        t = T - pd.DateOffset(months=h)
        train = R[R["T"] <= t - pd.DateOffset(months=L)]
        test = R[R["T"] == T]
        if len(train) < 60 or test.empty:
            continue
        te = test.copy()
        a_b = alpha if base_only else (alpha_base if alpha_base is not None else 0.0)
        mu_b, _ = _fit_predict(train, test, ["x_ar"], a_b)
        te["mu_glm_base"] = mu_b
        if not base_only:
            mu_a, _ = _fit_predict(train, test, ["x_ar"] + feature_cols, alpha)
            te["mu_glm_aug"] = mu_a
        rows.append(te)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_full(F, feats_all, name_map, feature_sets, label):
    """Selection on validation, then frozen evaluation on validation and test across L and h."""
    sel, base = select_config(F, feats_all, feature_sets, name_map)
    best = sel.sort_values("validation_mean_deviance").iloc[0]
    best_base = base.sort_values("validation_mean_deviance").iloc[0]
    fs = best["feature_set"].split("+")
    cols = [f"z_{k}" for k in fs]
    zc_all = [f"z_{k}" for k in name_map]
    preds = []
    for L in LAGS:
        for h in HORIZONS:
            # same rows for both models: require all z features of the selected set
            for period in ["validation", "test"]:
                p = evaluate(F, {k: feats_all[k] for k in name_map}, cols, period, L, h, best_base["alpha"], best["alpha"])
                if not p.empty:
                    preds.append(p)
    pred = pd.concat(preds, ignore_index=True)
    pred["config"] = label
    met = metrics(pred)
    met["config"] = label
    sel["config"], base["config"] = label, label
    chosen = pd.DataFrame([{"config": label, "feature_set": best["feature_set"], "alpha_augmented": best["alpha"],
                            "alpha_baseline": best_base["alpha"]}])
    return pred, met, sel, base, chosen


# ------------------------------------------------------------------ lead-lag association
def lead_lag(F, S, max_lead=6, B=500, seed=SEED):
    """Pooled Spearman correlation between nonfatal momentum at t and fatal momentum at t+k (k = 0..6)."""
    mF = _momentum(F)
    rng = np.random.default_rng(seed)
    rows = []
    for name, W in S.items():
        mS = _momentum(W)
        for k in range(0, max_lead + 1):
            a = mS.reindex(F.index)
            b = mF.shift(-k)
            st = pd.DataFrame({"x": a.stack(future_stack=True), "y": b.stack(future_stack=True)}).dropna()
            st = st[(st.index.get_level_values(0) >= "2021-03-01") & (st.index.get_level_values(0) <= "2024-12-01")]
            if len(st) < 100:
                continue
            r = stats.spearmanr(st["x"], st["y"])[0]
            codes, uniq = pd.factorize(st.index.get_level_values(1))
            groups = [np.where(codes == i)[0] for i in range(len(uniq))]
            boots = []
            for _ in range(B):
                pick = rng.integers(0, len(groups), len(groups))
                idx = np.concatenate([groups[i] for i in pick])
                boots.append(stats.spearmanr(st["x"].to_numpy()[idx], st["y"].to_numpy()[idx])[0])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"indicator": name, "lead_months": k, "spearman_rho": r, "ci_lo": lo, "ci_hi": hi, "n": len(st),
                         "n_jurisdictions": len(uniq)})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "earlywarning_lead_lag.csv", index=False)
    return out


# ------------------------------------------------------------------ runner
def run_earlywarning(mp, sysm, dx):
    """Primary analysis + sensitivity configurations. Returns dict of result tables (also written to CSV)."""
    res = {"pred": [], "met": [], "sel": [], "base": [], "chosen": [], "diff": []}
    configs = [
        ("primary: DOSE-SYS, cohort27", "cohort27", "sys"),
        ("sens: DOSE-SYS, all available jurisdictions", "all_available", "sys"),
        ("sens: DOSE-dx ED all-drug rate, cohort27", "cohort27", "dx"),
    ]
    sample_info = []
    for label, scope, source in configs:
        F, S, D = build_wide(mp, sysm, dx, scope)
        if source == "sys":
            feats, name_map, fsets = S, ["drug", "opioid", "stimulant"], FEATURE_SETS_SYS
        else:
            feats, name_map, fsets = D, ["ed_alldrug"], [("ed_alldrug",)]
        pred, met, sel, base, chosen = run_full(F, feats, name_map, fsets, label)
        for key, df in zip(["pred", "met", "sel", "base", "chosen"], [pred, met, sel, base, chosen]):
            res[key].append(df)
        for mb in ["glm_base", "persistence", "ma3"]:
            d = cluster_bootstrap_diff(pred, "glm_aug", mb)
            d["config"] = label
            res["diff"].append(d)
        sample_info.append({"config": label, "scope": scope, "source": source,
                            "n_jurisdictions_in_fatal_panel": F.shape[1],
                            "n_test_rows_h1_L0": int(((pred["period"] == "test") & (pred["L"] == 0) & (pred["h"] == 1)).sum()),
                            "n_validation_rows_h1_L0": int(((pred["period"] == "validation") & (pred["L"] == 0) & (pred["h"] == 1)).sum())})
    out = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
    out["sample_info"] = pd.DataFrame(sample_info)
    for k, df in out.items():
        df.to_csv(PROC / f"earlywarning_{k}.csv", index=False)
    F, S, D = build_wide(mp, sysm, dx, "cohort27")
    out["lead_lag"] = lead_lag(F, S)
    return out
