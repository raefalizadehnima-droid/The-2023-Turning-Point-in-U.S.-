"""Publication figures. Every figure is drawn from a DataFrame that is also written to CSV (figure source data)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp_
import numpy as np
import pandas as pd

from config import FIG, PROC, SUDORS_DRUGS

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False, "figure.dpi": 100})
COL = {"alldrug": "#222222", "opioids": "#1f77b4", "imfs": "#d62728", "stimulant": "#2ca02c", "meth": "#9467bd",
       "cocaine": "#ff7f0e", "heroin": "#8c564b", "rxopioids": "#17becf", "benzodiazepines": "#7f7f7f",
       "nonopioid_sedatives": "#bcbd22"}


def _save(fig, name, src: pd.DataFrame):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    src.to_csv(FIG / f"{name}_source_data.csv", index=False)
    plt.close(fig)


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axis("off")
    boxes = [
        (0.02, 0.70, "SUDORS dashboard file\nfatal; jurisdiction-year with\nmonthly counts; 2020-2024"),
        (0.02, 0.38, "DOSE-SYS file\nnonfatal ED; jurisdiction-month\nper 10,000 ED visits; 2019-2026"),
        (0.02, 0.06, "DOSE-dx (discharge) file\nnonfatal ED and inpatient counts;\n2018-2024"),
        (0.36, 0.52, "Audit and harmonisation\nsentinel codes -> missing\nde-duplicate mirror variables\nfixed jurisdiction cohorts"),
        (0.36, 0.14, "Quality control\nrecompute published totals\nsum checks, key uniqueness"),
        (0.70, 0.76, "RQ1-2  Trends, share screening\n(BH), change-point scans"),
        (0.70, 0.50, "RQ3  Rolling-origin forecasts\ntrain<=2022, valid 2023, test 2024"),
        (0.70, 0.24, "RQ4-5  Geography, cohort\ncomposition, sensitivity"),
    ]
    for x, y, t in boxes:
        w = 0.28
        ax.add_patch(mp_.FancyBboxPatch((x, y), w, 0.2, boxstyle="round,pad=0.01", fc="#f2f2f2", ec="#555555"))
        ax.text(x + w / 2, y + 0.1, t, ha="center", va="center", fontsize=7.5)
    for (x0, y0, x1, y1) in [(0.30, 0.80, 0.36, 0.64), (0.30, 0.48, 0.36, 0.60), (0.30, 0.16, 0.36, 0.56),
                             (0.50, 0.52, 0.50, 0.34), (0.64, 0.64, 0.70, 0.84), (0.64, 0.62, 0.70, 0.60), (0.64, 0.58, 0.70, 0.34)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    src = pd.DataFrame([{"box": t.replace("\n", " "), "x": x, "y": y} for x, y, t in boxes])
    _save(fig, "Figure1_workflow", src)


def fig2_trends(cm, ser):
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    rows = []
    for d in ["alldrug", "opioids", "imfs", "stimulant"]:
        ax[0].plot(cm["date"], cm[d], color=COL[d], lw=1.4, label=SUDORS_DRUGS[d])
        rows += [{"panel": "a", "series": d, "date": r.date, "value": getattr(r, d)} for r in cm.itertuples()]
    ax[0].set_ylabel("Overdose deaths per month"); ax[0].set_title("(a) SUDORS, fixed cohort of 27 jurisdictions")
    ax[0].set_ylim(0, None); ax[0].legend(fontsize=7)
    lab = {"drug": "All drugs", "opioid": "Opioids", "stimulant": "Stimulants"}
    for k, c in zip(["drug", "opioid", "stimulant"], ["#222222", "#1f77b4", "#2ca02c"]):
        s = ser[ser["indicator"] == k]
        ax[1].plot(s["date"], s["mean"], color=c, lw=1.4, label=lab[k])
        rows += [{"panel": "b", "series": k, "date": r.date, "value": r.mean} for r in s.itertuples()]
    ax[1].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-31"), color="#dddddd", lw=0)
    ax[1].set_ylabel("Suspected ED overdoses per 10,000 ED visits"); ax[1].set_ylim(0, None)
    ax[1].set_title("(b) DOSE-SYS, unweighted mean of fixed cohorts"); ax[1].legend(fontsize=7)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, "Figure2_temporal_trends", pd.DataFrame(rows))


def fig3_shares(ann, chg):
    drugs = [d for d in SUDORS_DRUGS if d != "alldrug"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    a20 = ann[ann.year == 2020].set_index("drug"); a24 = ann[ann.year == 2024].set_index("drug")
    order = sorted(drugs, key=lambda d: a24.loc[d, "share_pct"])
    y = np.arange(len(order))
    for i, d in enumerate(order):
        ax[0].plot([a20.loc[d, "share_pct"], a24.loc[d, "share_pct"]], [i, i], color="#999999", lw=1)
        ax[0].errorbar(a20.loc[d, "share_pct"], i, xerr=[[a20.loc[d, "share_pct"] - a20.loc[d, "share_lo_wilson"]], [a20.loc[d, "share_hi_wilson"] - a20.loc[d, "share_pct"]]],
                       fmt="o", color="#1f77b4", ms=4, lw=0.8)
        ax[0].errorbar(a24.loc[d, "share_pct"], i, xerr=[[a24.loc[d, "share_pct"] - a24.loc[d, "share_lo_wilson"]], [a24.loc[d, "share_hi_wilson"] - a24.loc[d, "share_pct"]]],
                       fmt="s", color="#d62728", ms=4, lw=0.8)
    ax[0].set_yticks(y); ax[0].set_yticklabels([SUDORS_DRUGS[d] for d in order], fontsize=7.5)
    ax[0].set_xlabel("% of all drug overdose deaths"); ax[0].set_title("(a) Share involving each class")
    ax[0].plot([], [], "o", color="#1f77b4", label="2020"); ax[0].plot([], [], "s", color="#d62728", label="2024"); ax[0].legend(fontsize=7, loc="lower right")
    c = chg[(chg.from_year == 2023) & (chg.to_year == 2024) & (chg.quantity == "count")].set_index("drug")
    for i, d in enumerate(order):
        r = c.loc[d]
        ax[1].errorbar(r.relative_change_pct, i, xerr=[[r.relative_change_pct - r.ci_lo_pct], [r.ci_hi_pct - r.relative_change_pct]], fmt="o", color="#222222", ms=4, lw=0.8)
    ax[1].axvline(0, color="#999999", lw=0.8); ax[1].set_xlabel("Change in deaths, 2023 to 2024 (%)")
    ax[1].set_title("(b) Counts fell for every class")
    fig.tight_layout()
    src = pd.concat([ann[ann.drug.isin(drugs)].assign(panel="a"), c.reset_index().assign(panel="b")], ignore_index=True)
    _save(fig, "Figure3_drug_distribution", src)


def fig4_changepoints(cm, det):
    keys = ["alldrug", "opioids", "imfs", "stimulant"]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
    rows = []
    for ax, d in zip(axs.ravel(), keys):
        D = det[d]
        fitted = np.exp(D["y"] - D["fit"]["resid"]) * cm["days_in_month"].to_numpy()
        ax.plot(cm["date"], cm[d], "o", ms=2.5, color="#555555", label="Observed")
        ax.plot(cm["date"], fitted, color=COL[d], lw=1.6, label="Segmented fit")
        for i, tau in enumerate(D["fit"]["taus"]):
            ax.axvline(cm["date"].iloc[tau], color=COL[d], ls="--", lw=0.8)
            if len(D["boots"]):
                lo, hi = np.percentile(D["boots"][:, i], [2.5, 97.5])
                ax.axvspan(cm["date"].iloc[int(lo)], cm["date"].iloc[int(hi)], color=COL[d], alpha=0.15, lw=0)
        ax.set_title(SUDORS_DRUGS[d]); ax.set_ylim(0, None)
        rows += [{"series": d, "date": r.date, "observed": getattr(r, d), "fitted": f} for r, f in zip(cm.itertuples(), fitted)]
        rows += [{"series": d, "date": cm["date"].iloc[t], "breakpoint_index": i + 1} for i, t in enumerate(D["fit"]["taus"])]
    axs[0, 0].legend(fontsize=7)
    for ax in axs[1]:
        ax.tick_params(axis="x", rotation=30)
    fig.supylabel("Deaths per month", fontsize=9)
    fig.tight_layout()
    _save(fig, "Figure4_changepoints", pd.DataFrame(rows))


def fig5_geography(sc):
    g = sc[sc.from_year == 2023].sort_values(["region", "pct_change"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    cols = {"Northeast": "#1f77b4", "Midwest": "#2ca02c", "South": "#ff7f0e", "West": "#9467bd"}
    for i, r in g.iterrows():
        ax.errorbar(r["pct_change"], i, xerr=[[r["pct_change"] - r["ci_lo_pct"]], [r["ci_hi_pct"] - r["pct_change"]]], fmt="o", ms=3.5, color=cols[r["region"]], lw=0.8)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g["abbr"], fontsize=7)
    ax.axvline(0, color="#999999", lw=0.8); ax.set_xlabel("Change in all-drug overdose deaths, 2023 to 2024 (%)\n(Poisson 95% intervals; ignore overdispersion)")
    for k, c in cols.items():
        ax.plot([], [], "o", color=c, label=k)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "Figure6_geographic_variation", g.drop(columns=["log_ratio", "se"]))


def fig6_earlywarning(diff, pred):
    d = diff[(diff.period == "test") & (diff.model_b == "glm_base") & (diff.config.str.startswith("primary"))]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3))
    cols = {0: "#1f77b4", 3: "#ff7f0e", 6: "#d62728"}
    for j, L in enumerate([0, 3, 6]):
        s = d[d.L == L]
        x = s.h.to_numpy() + (j - 1) * 0.12
        ax[0].errorbar(x, s.relative_diff_pct, yerr=[s.relative_diff_pct - s.rel_ci_lo_pct, s.rel_ci_hi_pct - s.relative_diff_pct],
                       fmt="o", color=cols[L], label=f"fatal data lag L = {L} mo", ms=4, lw=0.9)
    ax[0].axhline(0, color="#999999", lw=0.8); ax[0].set_xticks([1, 2, 3]); ax[0].set_xlabel("Forecast horizon h (months)")
    ax[0].set_ylabel("Change in mean Poisson deviance (%)\nvs fatal-history GLM (negative = better)")
    ax[0].set_title("(a) Added value of DOSE-SYS features, 2024 test"); ax[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    p = pred[(pred.config.str.startswith("primary")) & (pred.period == "test") & (pred.L == 3) & (pred.h == 3)]
    s = p.groupby("T").agg(observed=("y", "sum"), persistence=("mu_persistence", "sum"), glm_base=("mu_glm_base", "sum"), glm_aug=("mu_glm_aug", "sum")).reset_index()
    ax[1].plot(s["T"], s["observed"], "k-o", ms=3, label="Observed")
    ax[1].plot(s["T"], s["persistence"], color="#999999", label="Persistence")
    ax[1].plot(s["T"], s["glm_base"], color="#1f77b4", label="GLM, fatal history")
    ax[1].plot(s["T"], s["glm_aug"], color="#d62728", label="GLM + DOSE-SYS")
    ax[1].set_ylim(0, None); ax[1].set_ylabel("Deaths per month (sum of jurisdictions)"); ax[1].set_title("(b) Aggregate forecasts, L = 3, h = 3")
    ax[1].legend(fontsize=7); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, "Figure5_earlywarning_performance", pd.concat([d.assign(panel="a"), s.assign(panel="b")], ignore_index=True))


def fig7_robustness(sens, jk):
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    s = sens[sens.series.isin(["alldrug", "imfs", "stimulant", "opioids"])].copy()
    s["bp"] = pd.to_datetime(s["bp_month"] + "-01")
    specs = list(dict.fromkeys(s["specification"]))
    for i, sp in enumerate(specs):
        for d in ["alldrug", "opioids", "imfs", "stimulant"]:
            v = s[(s.specification == sp) & (s.series == d)]["bp"]
            if len(v):
                ax[0].plot(v.iloc[0], i, "o", color=COL[d], ms=4)
    for d in ["alldrug", "opioids", "imfs", "stimulant"]:
        ax[0].plot([], [], "o", color=COL[d], label=SUDORS_DRUGS[d])
    ax[0].set_yticks(range(len(specs))); ax[0].set_yticklabels(specs, fontsize=6.5); ax[0].invert_yaxis()
    import matplotlib.dates as mdates
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[0].tick_params(axis="x", labelsize=7)
    ax[0].set_xlabel("Estimated breakpoint (single-break model)"); ax[0].legend(fontsize=6.5, loc="lower left")
    ax[0].set_title("(a) Specification sensitivity")
    j = jk.copy()
    allm = sorted(j["bp_month"].unique())
    for k, d in enumerate(["alldrug", "imfs", "stimulant"]):
        v = j[j.series == d]["bp_month"].value_counts().reindex(allm).fillna(0)
        ax[1].bar(np.arange(len(allm)) + (k - 1) * 0.27, v.values, width=0.25, color=COL[d], label=SUDORS_DRUGS[d])
    ax[1].set_xticks(np.arange(len(allm)))
    ax[1].set_xticklabels(allm, rotation=45, fontsize=7)
    ax[1].set_ylabel("Leave-one-jurisdiction-out fits"); ax[1].set_title("(b) Jackknife"); ax[1].legend(fontsize=6.5)
    fig.tight_layout()
    _save(fig, "Figure7_robustness", pd.concat([s.drop(columns=["bp"]).assign(panel="a"), j.assign(panel="b")], ignore_index=True))
