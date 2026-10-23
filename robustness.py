"""RQ4/RQ5: geographic heterogeneity, cohort-composition sensitivity, leave-one-region-out."""
import numpy as np
import pandas as pd
from scipy import stats

from config import PROC, REGION


def state_changes(mp: pd.DataFrame) -> pd.DataFrame:
    """Per-jurisdiction 2023->2024 and 2020->2023 change in all-drug deaths (cohort27) + random-effects pooling."""
    c = mp[mp["in_cohort27"]]
    a = c.groupby(["abbr", "year"])["alldrug"].sum().unstack("year")
    rows = []
    for s, r in a.iterrows():
        for (y0, y1) in [(2023, 2024), (2020, 2023)]:
            lr = np.log(r[y1] / r[y0])
            se = np.sqrt(1 / r[y0] + 1 / r[y1])  # Poisson SE of log ratio (ignores overdispersion)
            rows.append({"abbr": s, "region": REGION.get(s, "Other"), "from_year": y0, "to_year": y1,
                         "deaths_from": int(r[y0]), "deaths_to": int(r[y1]),
                         "pct_change": 100 * (np.exp(lr) - 1),
                         "ci_lo_pct": 100 * (np.exp(lr - 1.96 * se) - 1), "ci_hi_pct": 100 * (np.exp(lr + 1.96 * se) - 1),
                         "log_ratio": lr, "se": se})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "geo_state_changes.csv", index=False)
    return out


def random_effects(sc: pd.DataFrame) -> pd.DataFrame:
    """DerSimonian-Laird pooling of log ratios; reports tau^2, I^2 and a 95% prediction interval."""
    rows = []
    for (y0, y1), g in sc.groupby(["from_year", "to_year"]):
        y, v = g["log_ratio"].to_numpy(), g["se"].to_numpy() ** 2
        w = 1 / v
        mu_f = (w * y).sum() / w.sum()
        Q = (w * (y - mu_f) ** 2).sum()
        k = len(y)
        tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        ws = 1 / (v + tau2)
        mu = (ws * y).sum() / ws.sum()
        se_mu = np.sqrt(1 / ws.sum())
        I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
        t = stats.t.ppf(0.975, k - 2)
        pi = (mu - t * np.sqrt(tau2 + se_mu ** 2), mu + t * np.sqrt(tau2 + se_mu ** 2))
        rows.append({"from_year": y0, "to_year": y1, "k_jurisdictions": k,
                     "pooled_pct_change": 100 * (np.exp(mu) - 1),
                     "ci_lo": 100 * (np.exp(mu - 1.96 * se_mu) - 1), "ci_hi": 100 * (np.exp(mu + 1.96 * se_mu) - 1),
                     "pred_lo": 100 * (np.exp(pi[0]) - 1), "pred_hi": 100 * (np.exp(pi[1]) - 1),
                     "tau2": tau2, "I2_pct": I2, "Q": Q,
                     "share_declined": float((g["pct_change"] < 0).mean()),
                     "n_ci_excluding_zero_declines": int((g["ci_hi_pct"] < 0).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "geo_random_effects.csv", index=False)
    return out


def composition_sensitivity(mp: pd.DataFrame, overall_pub: pd.DataFrame) -> pd.DataFrame:
    """How much does the 2023->2024 change in all-drug deaths depend on which jurisdictions are counted?"""
    a = mp.groupby(["abbr", "year"])["alldrug"].sum().unstack("year")
    rows = []

    def add(label, sel):
        x = a.loc[sel]
        rows.append({"definition": label, "n_jurisdictions": len(sel), "deaths_2023": int(x[2023].sum()),
                     "deaths_2024": int(x[2024].sum()), "pct_change_2023_2024": 100 * (x[2024].sum() / x[2023].sum() - 1)})
    add("Fixed cohort: all five years (primary)", a.dropna(subset=[2020, 2021, 2022, 2023, 2024]).index)
    add("Paired: present in both 2023 and 2024", a.dropna(subset=[2023, 2024]).index)
    add("Paired excluding cohort27", a.dropna(subset=[2023, 2024]).index.difference(a.dropna(subset=[2020, 2021, 2022, 2023, 2024]).index))
    o = overall_pub.set_index("data_year")["alldrug_deaths"]
    rows.append({"definition": "Published 'Overall' rows (membership changes: 42 vs 43 jurisdictions)", "n_jurisdictions": np.nan,
                 "deaths_2023": int(o[2023]), "deaths_2024": int(o[2024]),
                 "pct_change_2023_2024": 100 * (o[2024] / o[2023] - 1)})
    for reg in ["Northeast", "Midwest", "South", "West"]:
        coh = a.dropna(subset=[2020, 2021, 2022, 2023, 2024]).index
        add(f"Cohort27 excluding {reg}", [s for s in coh if REGION.get(s) != reg])
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "robust_composition.csv", index=False)
    return out


def region_summary(sc: pd.DataFrame) -> pd.DataFrame:
    g = sc[(sc["from_year"] == 2023)].groupby("region").agg(
        n=("abbr", "count"), deaths_2023=("deaths_from", "sum"), deaths_2024=("deaths_to", "sum"))
    g["pct_change"] = 100 * (g["deaths_2024"] / g["deaths_2023"] - 1)
    g = g.reset_index()
    g.to_csv(PROC / "geo_region_summary.csv", index=False)
    return g
