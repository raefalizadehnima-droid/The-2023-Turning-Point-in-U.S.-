"""Assemble the manuscript from computed result tables. Every number in the text is read from a CSV in
03_Processed_Data (or computed here from one); nothing is typed in by hand except wording."""
import json

import numpy as np
import pandas as pd

from config import DOC, FIG, PROC, SUDORS_DRUGS

P = lambda f: pd.read_csv(PROC / f)  # noqa: E731

cm = P("fatal_cohort27_monthly.csv")
ann = P("fatal_cohort27_annual.csv")
chg = P("fatal_cohort27_changes.csv")
tm = P("fatal_trend_models.csv")
scr = P("fatal_category_screening.csv")
cp = P("fatal_changepoints.csv").set_index("series")
cps = P("fatal_changepoint_sensitivity.csv")
jk = P("fatal_changepoint_jackknife.csv")
sys_ann = pd.read_csv(PROC / "dose_sys_pooled_annual_means.csv").set_index("indicator")
sys_jj = pd.read_csv(PROC / "dose_sys_pooled_janjul_means.csv").set_index("indicator")
sys_cp = P("dose_sys_changepoints.csv").set_index("indicator")
sys_coh = P("dose_sys_cohorts.csv").set_index("indicator")
sys_cov = P("dose_sys_indicator_coverage.csv").set_index("indicator")
sys_sc = P("dose_sys_state_changes.csv").set_index("indicator")
dxp = P("dose_dx_pooled_annual.csv")
conc = P("dose_sys_dx_concordance.csv")
ewd = P("earlywarning_diff.csv")
ewm = P("earlywarning_met.csv")
ewc = P("earlywarning_chosen.csv").set_index("config")
ewi = P("earlywarning_sample_info.csv").set_index("config")
ll = P("earlywarning_lead_lag.csv")
re_ = P("geo_random_effects.csv")
comp = P("robust_composition.csv")
reg = P("geo_region_summary.csv").set_index("region")
qc = P("qc_checks.csv")
inv = pd.read_csv(DOC / "data_inventory.csv")
flog = P("filter_log.csv")
PRIM = "primary: DOSE-SYS, cohort27"
SENS_ALL = "sens: DOSE-SYS, all available jurisdictions"
SENS_DX = "sens: DOSE-dx ED all-drug rate, cohort27"


def n(x, d=0):
    return f"{x:,.{d}f}"


def pct(x, d=1, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "\u2212") + "%"


def mon(s):
    return pd.Timestamp(s + "-01").strftime("%B %Y")


def C(drug, y0, y1, kind="count"):
    r = chg[(chg.drug == drug) & (chg.from_year == y0) & (chg.to_year == y1) & (chg.quantity == kind)].iloc[0]
    return r.relative_change_pct, r.ci_lo_pct, r.ci_hi_pct


def CI(t):
    e, lo, hi = t
    return f"{pct(e, 1, True)} (95% CI {pct(lo, 1, True)} to {pct(hi, 1, True)})"


def A(drug, year, col="deaths"):
    return ann[(ann.drug == drug) & (ann.year == year)][col].iloc[0]


def fmtq(q):
    return "<0.001" if q < 0.001 else f"{q:.3f}"


def QC(substr):
    r = qc[qc.check.str.contains(substr, regex=False)]
    return r.iloc[0]


# ----------------------------------------------------------------- key numbers
peak = cm.loc[cm.alldrug.idxmax()]
trough = cm.loc[cm.alldrug.idxmin()]
n_screen = len(scr)
n_cat_all = 108
rising_up = scr[scr.exploratory_label == "share rising 2022-24 (count up)"]
rising_down = scr[scr.exploratory_label == "share rising 2022-24 (count down)"]
falling = scr[scr.exploratory_label == "share falling 2022-24"]
none_ = scr[scr.exploratory_label == "no credible share change 2022-24"]
det_up = rising_up[rising_up.family == "detected (toxicology)"]
dmp = flog[flog.step.str.contains("mirror")].iloc[0]
test_prim = ewd[(ewd.period == "test") & (ewd.config == PRIM) & (ewd.model_b == "glm_base")].set_index(["L", "h"])
val_prim = ewd[(ewd.period == "validation") & (ewd.config == PRIM) & (ewd.model_b == "glm_base")].set_index(["L", "h"])
re_last = re_[re_.from_year == 2023].iloc[0]
re_first = re_[re_.from_year == 2020].iloc[0]
SEL = ewc.loc[PRIM]

REF_D = {
    "kariisa": "Kariisa M, O\u2019Donnell J, Kumar S, Mattson CL, Goldberger BA. Illicitly manufactured fentanyl\u2013involved overdose deaths with detected xylazine \u2014 United States, January 2019\u2013June 2022. MMWR Morb Mortal Wkly Rep. 2023;72(26):721\u2013727. doi:10.15585/mmwr.mm7226a4",
    "odonnell": "O\u2019Donnell J, et al. Trends in and characteristics of drug overdose deaths involving illicitly manufactured fentanyls \u2014 United States, 2019\u20132020. MMWR Morb Mortal Wkly Rep. 2021;70(50):1740\u20131746. doi:10.15585/mmwr.mm7050e3",
    "patton": "Patton T, Trillo-Park S, Swift B, B\u00f3rquez A. Early detection and prediction of non-fatal drug-related incidents and fatal overdose outbreaks using the Farrington algorithm. Addiction. 2025;120(2):266\u2013275. doi:10.1111/add.16674",
    "skinner": "Skinner A, Li Y, Hallowell BD, Pratty C, Goedel WC, Allen B, et al. Association of non-fatal overdose surveillance data with concurrent and future overdose deaths in Rhode Island. Am J Epidemiol. 2026;195(5):1475\u20131478. doi:10.1093/aje/kwag018",
    "killick": "Killick R, Fearnhead P, Eckley IA. Optimal detection of changepoints with a linear computational cost. J Am Stat Assoc. 2012;107(500):1590\u20131598. doi:10.1080/01621459.2012.737745",
    "bh": "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc Series B. 1995;57(1):289\u2013300.",
    "muggeo": "Muggeo VMR. Estimating regression models with unknown break-points. Stat Med. 2003;22(19):3055\u20133071. doi:10.1002/sim.1545",
    "dl": "DerSimonian R, Laird N. Meta-analysis in clinical trials. Control Clin Trials. 1986;7(3):177\u2013188.",
    "wilson": "Wilson EB. Probable inference, the law of succession, and statistical inference. J Am Stat Assoc. 1927;22(158):209\u2013212.",
    "sudors": "Centers for Disease Control and Prevention. State Unintentional Drug Overdose Reporting System (SUDORS) dashboard: fatal overdose data (downloadable workbook \u2018SUDORS-Fatal-Overdose-Data.xlsx\u2019, including its Info Tab and Data Dictionary).",
    "dose": "Centers for Disease Control and Prevention. Drug Overdose Surveillance and Epidemiology (DOSE) dashboards: DOSE-SYS download of 28 September 2026 and DOSE discharge-data (\u2018dx\u2019) dashboard download of 5 August 2026 (workbooks \u2018DOSE_SYS_Download_09-28-2026.xlsx\u2019 and \u2018DOSE_dx_Dashboard_08052026.xlsx\u2019, including the DOSE-SYS Data Dictionary).",
    "garnett": "Garnett MF, Mini\u00f1o AM. Drug overdose deaths in the United States, 2003\u20132023. NCHS Data Brief No. 522. Hyattsville, MD: National Center for Health Statistics; December 2024.",
    "spencer": "Spencer MR, Mini\u00f1o AM, Garnett MF. Co-involvement of opioids in drug overdose deaths involving cocaine and psychostimulants, 2011\u20132021. NCHS Data Brief No. 474. Hyattsville, MD: National Center for Health Statistics; July 2023.",
    "vangelov": "Vangelov K, Humphreys K, Caulkins JP, Pollack H, Pardo B, Reuter P. Did the illicit fentanyl trade experience a supply shock? Science. 2026;391(6781):134. doi:10.1126/science.aea6130",
    "pickens": "Pickens CM, et al. Trends in suspected fentanyl-involved nonfatal overdose emergency department visits, by age group, sex, and race and ethnicity \u2014 United States, October 2020\u2013March 2024. MMWR Morb Mortal Wkly Rep. 2025;74(16):282\u2013287. doi:10.15585/mmwr.mm7416a2",
    "khezri": "Khezri M, Rouhani S, Lindenfeld Z, Joss K, Huang M, Bather JR. Proportion of fentanyl reports in illicit drug seizures and nonfatal overdose emergency department visits in the United States, 2021\u20132024. Int J Drug Policy. 2026;155:105382. doi:10.1016/j.drugpo.2026.105382",
    "huo": "Huo S, London K, Murphy L, et al. Notes from the Field: Suspected medetomidine withdrawal syndrome among fentanyl-exposed patients \u2014 Philadelphia, Pennsylvania, September 2024\u2013January 2025. MMWR Morb Mortal Wkly Rep. 2025;74(15):266\u2013268. doi:10.15585/mmwr.mm7415a2",
    "tanz": "Tanz LJ, et al. Drug overdose deaths involving stimulants \u2014 United States, January 2018\u2013June 2024. MMWR Morb Mortal Wkly Rep. 2025;74(32). doi:10.15585/mmwr.mm7432a1",
    "cano": "Cano M, Daniulaityte R, Marsiglia F. Xylazine in overdose deaths and forensic drug reports in US states, 2019\u20132022. JAMA Netw Open. 2024;7(1):e2350630. doi:10.1001/jamanetworkopen.2023.50630",
    "rodda": "Rodda LN. The surge of bromazolam-related fatalities replacing other novel designer benzodiazepines-related fatalities in San Francisco. Addiction. 2024;119:1487.",
    "falise": "Falise AM, Sutphin A, Culbreth R, Aldy K, Burkhart K, Brent J, et al. Detected substances among patients with confirmed bromazolam exposure who present to the emergency department following a suspected opioid and/or stimulant-related non-fatal overdose. Drug Alcohol Depend. 2025 (published July 2025).",
    "mortman": "Mortman R, Durbhakula S, Emerick T. Nitazenes: are pain physicians aware of the risks? Pain Med. 2025:pnaf127. doi:10.1093/pm/pnaf127",
    "bauer": "Bauer C, Zhang K, Li W, Bernson D, Dammann O, LaRochelle MR, et al. Small area forecasting of opioid-related mortality: Bayesian spatiotemporal dynamic modeling approach. JMIR Public Health Surveill. 2023;9:e41450. doi:10.2196/41450",
}


def number_citations(blocks):
    """Replace [K:key,key] markers by [n] numbers in order of first appearance; return ordered reference list."""
    import re
    order = []
    pat = re.compile(r"\[K:([a-z0-9_,]+)\]")

    def sub(mt):
        nums = []
        for k in mt.group(1).split(","):
            if k not in REF_D:
                raise KeyError(f"Unknown reference key {k}")
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(set(nums))
        return "[" + ", ".join(str(n_) for n_ in nums) + "]"
    for b in blocks:
        if b["type"] in ("p", "title") and "text" in b:
            b["text"] = pat.sub(sub, b["text"])
    unused = [k for k in REF_D if k not in order]
    if unused:
        raise SystemExit(f"References defined but never cited: {unused}")
    return [REF_D[k] for k in order]


def build():
    B = []
    h1 = lambda t: B.append({"type": "h1", "text": t})  # noqa: E731
    h2 = lambda t: B.append({"type": "h2", "text": t})  # noqa: E731
    p = lambda t: B.append({"type": "p", "text": t})  # noqa: E731

    e_all_20_24 = C("alldrug", 2020, 2024)
    e_all_23_24 = C("alldrug", 2023, 2024)

    # ------------------------------------------------------------ title block
    B.append({"type": "title", "text": "Shifting Drug Patterns in U.S. Overdose Deaths, 2020\u20132024: Change-Point Evidence from State Surveillance Data and the Limited Early-Warning Value of Nonfatal Emergency Indicators"})
    B.append({"type": "p", "text": "[Author names and affiliations to be completed by the authors]", "italic": True})

    # ------------------------------------------------------------ abstract
    h1("Abstract")
    bp = cp.loc["alldrug"]
    _l = test_prim.reset_index(); _l = _l[_l.L > 0].relative_diff_pct.abs(); lag_lo, lag_hi = _l.min(), _l.max()
    p(f"Background: Overdose deaths in the United States rose to a peak and then fell sharply, while the drug mix changed. Whether nonfatal emergency indicators can warn of such shifts ahead of mortality data is uncertain. "
      f"Methods: We analysed three public CDC aggregate workbooks: the SUDORS fatal-overdose dashboard (2020\u20132024), DOSE-SYS syndromic emergency department indicators (January 2019 to July 2026) and DOSE discharge data (2018\u20132024). After a documented audit, we fixed a cohort of {int(QC('Jurisdictions included in all 5').result)} jurisdictions with complete SUDORS reporting. We fitted segmented regressions with bootstrap intervals, screened {n_screen} drug categories for changes in their share of deaths with Benjamini\u2013Hochberg control, and tested whether DOSE-SYS momentum improves chronological, rolling-origin forecasts of monthly deaths in 2024 beyond fatal-history baselines. "
      f"Results: Cohort deaths fell from {n(peak.alldrug)} in {mon(peak.date[:7])}, the highest month, to {n(trough.alldrug)} in {mon(trough.date[:7])}; annual deaths were {n(A('alldrug',2023))} in 2023 and {n(A('alldrug',2024))} in 2024 ({CI(e_all_23_24)}). The estimated turning point was {mon(bp.bp2_month)} (95% bootstrap interval {mon(bp.bp2_boot95_lo)} to {mon(bp.bp2_boot95_hi)}), and the estimate moved by no more than a few months across specifications. "
      f"Counts fell for every drug class in 2024, yet the share of deaths involving stimulants rose from {A('stimulant',2020,'share_pct'):.1f}% to {A('stimulant',2024,'share_pct'):.1f}%, largely because opioid-involved deaths fell faster. Among toxicology-detected substances, {len(det_up)} showed rising shares together with rising counts in 2024, including bromazolam, carfentanil and nitazenes. "
      f"Adding DOSE-SYS features gave no reliable gain when fatal history was current; with fatal data lagged by three to six months the mean Poisson deviance was lower by roughly {lag_lo:.0f}\u2013{lag_hi:.0f}% in 2024, but intervals were wide and the gain did not appear in the 2023 validation year. "
      f"Conclusions: The fixed-cohort data support a well-timed mid-2023 downturn and a changing drug mix, but they do not show that nonfatal indicators provide dependable early warning. Results describe participating jurisdictions only and are not nationally representative.")
    p("Keywords: overdose surveillance; SUDORS; DOSE; change-point detection; polysubstance involvement; forecasting; reporting delay; fentanyl; stimulants; nitazenes", )

    # ------------------------------------------------------------ 1 Introduction
    h1("1. Introduction")
    p("Public-health agencies track overdose through two largely separate streams. Medical examiner and coroner systems yield detailed toxicology on deaths but report slowly, whereas emergency department systems report quickly and describe nonfatal events. The CDC State Unintentional Drug Overdose Reporting System (SUDORS) assembles death-certificate, medical examiner and toxicology information for unintentional and undetermined-intent overdose deaths in funded jurisdictions [K:sudors], and the Drug Overdose Surveillance and Epidemiology (DOSE) system collects nonfatal emergency department and hospital data from health departments [K:dose]. Both streams are published through dashboards from which summary workbooks can be downloaded.")
    p("National mortality has recently turned. The age-adjusted rate of drug overdose deaths declined 4.0% between 2022 and 2023 after rising across most years since 2003 [K:garnett], and a disruption of the illicit fentanyl trade beginning in 2023 has been proposed as an important contributor to the later fall in fentanyl-involved deaths [K:vangelov]. Counts of the kind analysed here cannot settle why deaths fell. They can, however, show when the fall began, which drug categories carried it, and whether it was uniform across reporting jurisdictions.")
    p("SUDORS has documented how the supply changed beneath the headline totals. Deaths involving illicitly manufactured fentanyls increased sharply in several regions during 2019\u20132020, and about four in ten also involved a stimulant [K:odonnell]. Detection of the veterinary sedative xylazine in fentanyl-involved deaths rose from 2.9% to 10.9% of such deaths between January 2019 and June 2022 in 21 jurisdictions [K:kariisa]. Across SUDORS data from January 2021 to June 2024, 59.0% of deaths involved stimulants and nearly three quarters of those also involved opioids [K:tanz], in line with earlier vital-statistics evidence of opioid co-involvement in cocaine and psychostimulant deaths [K:spencer]. State-level reports for 2019\u20132022 found xylazine in forensic reports and deaths concentrated in eastern states but spreading across the country [K:cano]. These reports describe particular substances over particular windows. What is less often examined is how a whole set of substances and combinations moved together once total deaths began to fall, and how robust such statements are to the choice of jurisdictions.")
    p("A second question is practical. Nonfatal indicators have followed the same turn: national syndromic data showed fentanyl-involved overdose emergency department visit rates rising 8.7% per quarter from late 2020 to the third quarter of 2023 and then falling 11% per quarter through early 2024 [K:pickens], and state-level DOSE-SYS rates have been linked to the share of fentanyl in drug seizures [K:khezri]. Because mortality data are delayed, several groups have asked whether nonfatal indicators can serve as an early signal. A county-level study in California and Florida found that aberration-detection alerts in nonfatal data could help identify sustained increases in fatal overdose, with the benefit depending on the method used [K:patton]. A study in Rhode Island linked emergency medical services records to SUDORS at census block group level and found strong concurrent associations that weakened when nonfatal data were lagged or aggregated to wider areas and longer intervals [K:skinner]. Both studies work within one or two states, as does much forecasting work, such as small-area Bayesian models for opioid mortality in Massachusetts [K:bauer]. We did not identify, in the focused search described in Section 2.6, a chronological out-of-sample evaluation that uses state-level DOSE syndromic indicators to predict SUDORS monthly counts under explicit assumptions about reporting delay.")
    p("The aims of this paper are narrow. First, we describe how fatal overdose outcomes and drug involvement changed in a fixed set of SUDORS jurisdictions between 2020 and 2024, and we date the turning point with uncertainty. Second, we screen drug categories for changes in their share of deaths while separating share from count, since the two can move in opposite directions. Third, we test whether DOSE-SYS indicators improve forecasts of fatal counts beyond transparent baselines. Fourth, we examine how far the findings depend on jurisdiction composition, reporting completeness and modelling choices. The analyses are exploratory except where stated, and they use aggregate data only.")

    # ------------------------------------------------------------ 2 Methods
    h1("2. Materials and Methods")
    h2("2.1 Data sources and observation units")
    p("We analysed three Excel workbooks downloaded from CDC dashboards (Table 1). None contains individual-level records. The SUDORS workbook holds one row per jurisdiction and data year (2020\u20132024) with annual and monthly counts of deaths by drug class, percentages, age-adjusted rates and pre-computed combination counts, together with composite \u2018Overall\u2019 rows and trend tabs. DOSE-SYS reports, by jurisdiction and month, suspected nonfatal overdose emergency department visits per 10,000 emergency department visits for eight drug indicators; it does not publish visit counts. The DOSE discharge workbook (\u2018dx\u2019) reports counts and crude rates per 100,000 population for emergency department and inpatient discharge data. File identities and SHA-256 checksums are in the data inventory. The download dates of the SUDORS file are not stated inside it; the DOSE files carry dates in their names.")
    p("Where the data are aggregate we kept them aggregate. All statements concern jurisdictions and months, not individuals, and none of the analyses supports inference about individual behaviour.")
    B.append({"type": "figure", "path": str(FIG / "Figure1_workflow.png"), "caption": "Figure 1. Study workflow: three public CDC workbooks, audit and harmonisation, quality control, and the analyses addressing each research question. Source data: Figure1_workflow_source_data.csv.", "width": 560})
    h2("2.2 Audit, harmonisation and cohort definition")
    p(f"Before analysis we inventoried every sheet, checked key uniqueness, tested internal arithmetic (for example, whether twelve monthly counts sum to the annual total) and recomputed published totals. Sentinel values were converted to missing and never recoded to zero: 9999 in SUDORS, the symbols \u2018*\u2019, \u2018\u00a7\u2019 and \u2018\u00b6\u2019 in DOSE-SYS, and \u2018suppressed\u2019 or empty cells in DOSE-dx. SUDORS stores each pairwise combination twice (for example imfs_heroin and heroin_imfs); we identified mirror pairs by variable name, confirmed identical values, and kept one copy ({int(dmp.rows_before)} combination variables reduced to {int(dmp.rows_after)}). Value equality alone was not used, because columns made entirely of 9999 would otherwise have been merged.")
    p("SUDORS includes a jurisdiction in a given year only if it reported all overdose deaths and had circumstance information for at least 75% of them, so the composite \u2018Overall\u2019 rows change membership from year to year and are not comparable across years. Our primary population is therefore the fixed cohort of jurisdictions included in every year from 2020 to 2024. Category-specific analyses further required a valid (non-9999) count in all five years for that category. DOSE-SYS pooled series use fixed cohorts of jurisdictions with a numeric value in every month from January 2021 to July 2026, and are unweighted means of jurisdiction values because visit counts are not published.")
    h2("2.3 Trend estimation, share screening and change points")
    p("Annual relative changes in cohort deaths and in the share of all drug overdose deaths involving each class were estimated from cohort totals, with 95% percentile intervals from 2,000 bootstrap resamples of jurisdictions. Shares use Wilson intervals conditional on the all-drug total [K:wilson]. Log-linear trends in monthly cohort counts were fitted as quasi-Poisson regressions with an offset for days per month, one annual harmonic and Newey\u2013West standard errors (six lags); a negative binomial fit served as a sensitivity check. Residual autocorrelation and dispersion were examined.")
    p(f"For the screening of emerging patterns, each eligible category (single classes, de-duplicated combinations, exclusive categories and drugs detected by toxicology) was modelled with a binomial regression of deaths involving the category among all drug overdose deaths on centred year, using jurisdiction-years and cluster-robust standard errors by jurisdiction. Two slope tests were run per category, over 2020\u20132024 and over 2022\u20132024, and the Benjamini\u2013Hochberg procedure [K:bh] was applied across all {n_screen} eligible categories separately for each window. A category was labelled as having a rising share only as a descriptive device, and its count change from 2023 to 2024 was always reported alongside, since an increase in share does not imply an increase in deaths.")
    p("Change points were estimated for monthly cohort deaths per day on the log scale with a continuous piecewise-linear (joinpoint-type) model [K:muggeo] including an annual harmonic. We searched all breakpoint locations exhaustively with a minimum segment of six months, chose zero, one or two breakpoints by the Bayesian information criterion, and obtained 95% intervals from a moving-block residual bootstrap (blocks of six months, 300 replicates), plus profile intervals for single-break models. As an alternative family we applied PELT [K:killick] and binary segmentation to the seasonally adjusted series. A breakpoint within one month of the admissible limit was flagged as a boundary solution. We treat detected changes as statistical features of the series; they are not evidence about causes.")
    h2("2.4 Early-warning evaluation")
    p("The forecasting target was the number of all drug overdose deaths in a jurisdiction in month T = t + h, for horizons h of 1, 2 and 3 months, where t is the month at which the nonfatal indicator is known. To represent reporting delay we considered three scenarios in which fatal history is available only through month t \u2212 L, with L = 0, 3 or 6. Baselines were persistence, a three-month moving average, a seasonal-naive forecast, and a Poisson regression using fatal history only (log ratio of the latest count to the three-month mean, with the three-month mean as exposure). The augmented model added DOSE-SYS momentum, defined as the log ratio of the mean of the latest three months to the mean of the preceding three months, for all-drug, opioid and stimulant indicators. Rows with a zero three-month mean had no defined log ratio and were excluded rather than altered.")
    p("Targets were split chronologically: training up to December 2022, validation in 2023 and test in 2024. For each evaluation month the model was refitted using only rows whose targets had been observed at the forecast origin. The feature set and ridge penalty were chosen on the validation year alone, using the Poisson deviance averaged over horizons at L = 0, and then frozen; the test year was evaluated once. Performance was summarised by mean absolute error, root mean squared error and mean Poisson deviance, with 80% and 95% negative-binomial predictive interval coverage for the regression models. Differences in mean deviance were given with 95% intervals from 2,000 bootstrap resamples of jurisdictions. Sensitivity analyses used all jurisdictions with usable data and replaced DOSE-SYS by the DOSE-dx emergency department rate. Lead\u2013lag associations between nonfatal and fatal momentum were summarised with pooled Spearman correlations and jurisdiction-cluster bootstrap intervals.")
    h2("2.5 Geographic heterogeneity and robustness")
    p("Jurisdiction-specific changes in deaths were pooled with a DerSimonian\u2013Laird random-effects model [K:dl], reporting heterogeneity and a 95% prediction interval. We compared the 2023 to 2024 change across alternative definitions of the jurisdiction set and with each census region removed in turn. Breakpoint sensitivity covered minimum segment length, truncation of the final months, exclusion of 2020, a Poisson deviance criterion, and leave-one-jurisdiction-out refits.")
    h2("2.6 Literature search, reproducibility and ethics")
    p("The literature review was focused rather than systematic. We searched for recent work on nonfatal indicators and later overdose deaths, SUDORS reports on fentanyl, stimulants and xylazine, and DOSE data comparisons, and checked the bibliographic details of the cited overdose-surveillance papers against publisher, PubMed Central, CDC or institutional repository pages during manuscript preparation. Methodological references (changepoint detection, segmented regression, false discovery rate, random-effects pooling and the Wilson interval) are standard citations given from the authors\u2019 knowledge and flagged for verification in the project documentation. All analyses were run in Python with fixed random seeds from a single entry point (main.py); software versions are recorded with the code.")

    # ------------------------------------------------------------ 3 Results
    h1("3. Results")
    h2("3.1 Data audit")
    fm = int(QC("Duplicate (jurisdiction, data_year)").result)
    p(f"All internal checks passed (Table 2). SUDORS contained {int(QC('Rows in Annual Data').result)} rows, including five composite rows, with {int(QC('Jurisdictions ever included').result)} jurisdictions included in at least one year and {int(QC('Jurisdictions included in all 5').result)} in all five. Monthly counts summed exactly to annual totals for every drug class, and our recomputed cohort totals matched the dashboard\u2019s published fixed-cohort totals for each year. The sentinel 9999 appeared in {n(int(QC('Cells equal to 9999').result))} cells, all outside the monthly death counts; summing them in cohort aggregates would have produced absurd totals, which confirms the need to treat them as missing. The published percentages for toxicology-detected drugs use a denominator that is not in the workbook, so our shares for those drugs divide by all drug overdose deaths and are conservative relative to the dashboard.")
    cov = sys_cov
    p(f"In DOSE-SYS, {cov.loc['drug','jurisdictions_with_any_value']:.0f} jurisdictions contributed numeric all-drug values from January 2021 onward, but only {cov.loc['drug','jurisdictions_complete_all_months']:.0f} had a complete series; for fentanyl, heroin, cocaine, methamphetamine and benzodiazepine indicators only 3 to 10 jurisdictions were complete (Table 2), so we limited pooled and predictive analyses to the all-drug, opioid and stimulant indicators. No numeric values exist for March to August 2020, when CDC excluded the months. The discharge workbook has no data dictionary, and the meaning of empty cells in its fentanyl and methamphetamine counts is undocumented; we treated them as unavailable. Monthly discharge counts summed exactly to annual counts, and the number of submitting jurisdictions agreed with the submission-count sheet.")
    B.append({"type": "table", "title": "Table 1. Datasets, observation levels and coverage.", "key": "T1"})
    B.append({"type": "table", "title": "Table 2. Data-quality assessment and missingness (selected checks).", "key": "T2"})

    h2("3.2 Temporal change in fatal overdose outcomes")
    p(f"In the 27-jurisdiction cohort, deaths numbered {n(A('alldrug',2020))} in 2020, {n(A('alldrug',2021))} in 2021, {n(A('alldrug',2022))} in 2022, {n(A('alldrug',2023))} in 2023 and {n(A('alldrug',2024))} in 2024 (Table 3; Figure 2a). Monthly deaths peaked at {n(peak.alldrug)} in {mon(peak.date[:7])} and reached a low of {n(trough.alldrug)} in {mon(trough.date[:7])}. Relative to 2023, deaths in 2024 changed by {CI(e_all_23_24)}; relative to 2020 the change was {CI(e_all_20_24)}. These two estimates reproduce the dashboard\u2019s published first-versus-last-year figure for this cohort.")
    p(f"The decline was shared by every drug class (Figure 3b). Deaths involving any opioid fell by {pct(abs(C('opioids',2023,2024)[0]))} between 2023 and 2024, those involving illicitly manufactured fentanyls by {pct(abs(C('imfs',2023,2024)[0]))}, stimulants by {pct(abs(C('stimulant',2023,2024)[0]))}, cocaine by {pct(abs(C('cocaine',2023,2024)[0]))} and methamphetamine by {pct(abs(C('meth',2023,2024)[0]))}. Relative to 2020, however, stimulant deaths were {CI(C('stimulant',2020,2024))}, cocaine {CI(C('cocaine',2020,2024))} and methamphetamine {CI(C('meth',2020,2024))}, while opioid deaths were {CI(C('opioids',2020,2024))} and heroin deaths {CI(C('heroin',2020,2024))}.")
    p(f"The composition of deaths changed accordingly. The share involving any opioid fell from {A('opioids',2020,'share_pct'):.1f}% to {A('opioids',2024,'share_pct'):.1f}%, illicitly manufactured fentanyls from {A('imfs',2020,'share_pct'):.1f}% (and {A('imfs',2023,'share_pct'):.1f}% in 2023) to {A('imfs',2024,'share_pct'):.1f}%, and heroin from {A('heroin',2020,'share_pct'):.1f}% to {A('heroin',2024,'share_pct'):.1f}%. The share involving any stimulant rose from {A('stimulant',2020,'share_pct'):.1f}% to {A('stimulant',2024,'share_pct'):.1f}%, methamphetamine from {A('meth',2020,'share_pct'):.1f}% to {A('meth',2024,'share_pct'):.1f}%, and cocaine from {A('cocaine',2020,'share_pct'):.1f}% to {A('cocaine',2024,'share_pct'):.1f}%.")
    ta = tm[(tm.drug == "alldrug") & (tm.window == "2020-01..2024-12")].iloc[0]
    p(f"A single log-linear slope described these series poorly. For all-drug deaths the 2020\u20132024 slope was {pct(ta.annual_pct_change,1,True)} per year (95% CI {pct(ta.ci_lo_pct,1,True)} to {pct(ta.ci_hi_pct,1,True)}), yet residuals were strongly autocorrelated (Ljung\u2013Box p < 0.001, Durbin\u2013Watson {ta.durbin_watson:.2f}) and overdispersed (Pearson dispersion {ta.pearson_dispersion:.0f}). Series that rise, plateau and fall are not well summarised by one slope, and the full model table is in the Supplementary Materials. This motivated the segmented analysis below.")
    B.append({"type": "table", "title": "Table 3. Fatal overdose deaths and share of all drug overdose deaths by drug class, fixed cohort of 27 SUDORS jurisdictions.", "key": "T3"})
    B.append({"type": "figure", "path": str(FIG / "Figure2_temporal_trends.png"), "caption": "Figure 2. Temporal trends. (a) Monthly overdose deaths in the 27-jurisdiction SUDORS cohort for selected drug classes. (b) DOSE-SYS suspected nonfatal overdose emergency department visits per 10,000 visits, as unweighted means over fixed cohorts of jurisdictions (42 for all drugs, 38 for opioids, 24 for stimulants); the grey band marks March\u2013August 2020, for which CDC published no values. Source data: Figure2_temporal_trends_source_data.csv.", "width": 620})
    B.append({"type": "figure", "path": str(FIG / "Figure3_drug_distribution.png"), "caption": "Figure 3. (a) Share of all drug overdose deaths involving each drug class in 2020 and 2024, with Wilson 95% intervals conditional on the all-drug total. Classes overlap, so shares do not sum to 100%. (b) Relative change in deaths from 2023 to 2024 with bootstrap 95% intervals (27 jurisdictions resampled). Source data: Figure3_drug_distribution_source_data.csv.", "width": 620})

    h2("3.3 Change points")
    kk = cp.loc["alldrug"]
    p(f"For all-drug deaths the information criterion favoured two breakpoints (BIC {kk.bic_k2:.1f} versus {kk.bic_k1:.1f} for one and {kk.bic_k0:.1f} for none; Table 4, Figure 4). The first lay at {mon(kk.bp1_month)}, at the lower limit of the admissible range, and the bootstrap interval ran from {mon(kk.bp1_boot95_lo)} to {mon(kk.bp1_boot95_hi)}. We therefore regard it as a boundary solution that mainly reflects the steep rise in the first months of the series, which our data cannot resolve because the series begins in January 2020. The second breakpoint is better identified: {mon(kk.bp2_month)} (95% bootstrap interval {mon(kk.bp2_boot95_lo)} to {mon(kk.bp2_boot95_hi)}). After it, deaths declined at an estimated {pct(abs(kk.segment3_annual_pct),0)} per year, compared with a near-flat slope of {pct(kk.segment2_annual_pct,1,True)} per year in the preceding segment.")
    ki, ks, ko, kc, kmt = cp.loc["imfs"], cp.loc["stimulant"], cp.loc["opioids"], cp.loc["cocaine"], cp.loc["meth"]
    p(f"The downturn was dated within a few months across classes: opioids {mon(ko.bp2_month)}, illicitly manufactured fentanyls {mon(ki.bp2_month)}, methamphetamine {mon(kmt.bp2_month)} and stimulants {mon(ks.bp2_month)}, with cocaine fitting a single break at {mon(kc.bp1_month)} (profile interval {mon(kc.bp1_profile_lo)} to {mon(kc.bp1_profile_hi)}). Intervals overlap substantially, so the data do not support ranking one class as turning first. Heroin showed a second breakpoint at {mon(cp.loc['heroin'].bp2_month)}, within one month of the admissible limit, and its final segment rests on a handful of months; we do not interpret it. PELT and binary segmentation, which detect level shifts and are sensitive to the penalty, returned between one and six breakpoints for the same series depending on the penalty (Supplementary Materials), a reminder that the location of a break is more reliable than the number of breaks.")
    B.append({"type": "table", "title": "Table 4. Change-point estimates for monthly deaths per day (log-scale piecewise-linear model).", "key": "T5"})
    B.append({"type": "figure", "path": str(FIG / "Figure4_changepoints.png"), "caption": "Figure 4. Observed monthly deaths and segmented fits for four series in the 27-jurisdiction cohort. Dashed lines mark estimated breakpoints; shaded bands show 95% moving-block bootstrap intervals. The first breakpoint in most panels lies at the boundary of the search range and should not be read as a distinct event. Source data: Figure4_changepoints_source_data.csv.", "width": 560})

    h2("3.4 Screening for emerging patterns")
    p(f"Of {n_cat_all} categories derivable from the SUDORS workbook, {n_screen} met the eligibility rule (valid counts in all five years for at least 10 jurisdictions and at least 100 deaths in 2024); the remainder, mostly combinations involving nitazenes, MDMA or cathinones, were unavailable in the cohort. Over 2022\u20132024, {len(rising_up)+len(rising_down)} categories showed a rising share with Benjamini\u2013Hochberg q < 0.05, {len(falling)} showed a falling share and {len(none_)} showed no credible change. Of the rising-share categories, {len(rising_down)} had fewer deaths in 2024 than in 2023. Their share rose mechanically because opioid-involved deaths fell faster than total deaths, and these should not be described as growing problems.")
    names = {"adulterants_sedatives": "adulterants and sedatives", "illegal_benzodiazepines": "illegal benzodiazepines", "other_illegal_opioids": "other illegal opioids", "mdma_mda": "MDMA/MDA"}
    dl = []
    for r in det_up.sort_values("count_change_2023_2024_pct", ascending=False).itertuples():
        dl.append(f"{names.get(r.category, r.category)} ({pct(r.count_change_2023_2024_pct,0,True)})")
    p(f"{len(det_up)} rising-share detected substances also had higher counts in 2024 than in 2023 (change in deaths 2023 to 2024 in parentheses): {', '.join(dl)}. Three features temper this. First, the counts are small for several substances (for example {int(scr[scr.category=='nitazenes'].deaths_2024.iloc[0])} deaths with nitazenes in 2024 and {int(scr[scr.category=='carfentanil'].deaths_2024.iloc[0])} with carfentanil, in a cohort of 26 jurisdictions). Second, detection depends on what laboratories test for, so rising detection may partly reflect expanded panels; nitazenes in particular can escape routine toxicology screens [K:mortman]. Third, patterns were not monotone: carfentanil was detected in {int(scr[scr.category=='carfentanil'].deaths_2020.iloc[0])} deaths in 2020 and {int(scr[scr.category=='carfentanil'].deaths_2022.iloc[0])} in 2022 before reaching {int(scr[scr.category=='carfentanil'].deaths_2024.iloc[0])} in 2024, and parafluorofentanyl rose to {scr[scr.category=='parafluorofentanyl'].share_2022_pct.iloc[0]:.1f}% of deaths in 2022 and then fell (q = {fmtq(scr[scr.category=='parafluorofentanyl'].recent_q_bh.iloc[0])} for 2022\u20132024 and {fmtq(scr[scr.category=='parafluorofentanyl'].full_q_bh.iloc[0])} for the full period). A recent appearance alone, therefore, is not enough to call a substance emerging; the full-period and recent slopes can disagree. The categories showing increases in both share and counts are listed in Table 5.")
    B.append({"type": "table", "title": "Table 5. Categories whose share of deaths rose over 2022\u20132024 (Benjamini\u2013Hochberg q < 0.05) while counts rose from 2023 to 2024.", "key": "T4"})

    h2("3.5 Nonfatal indicators")
    d = sys_ann.loc["drug"]; o = sys_ann.loc["opioid"]; s_ = sys_ann.loc["stimulant"]
    p(f"Pooled DOSE-SYS indicators fell steadily after 2021. The unweighted mean all-drug indicator was {d['2021']:.1f} suspected overdose visits per 10,000 emergency department visits in 2021 and {d['2025']:.1f} in 2025, the opioid indicator {o['2021']:.1f} and {o['2025']:.1f}, and the stimulant indicator {s_['2021']:.1f} and {s_['2025']:.1f} (Figure 2b). Comparing January to July only, the all-drug mean was {sys_jj.loc['drug','2025']:.1f} in 2025 and {sys_jj.loc['drug','2026']:.1f} in 2026. Between 2023 and 2025, {sys_sc.loc['drug','share_declined']*100:.0f}% of {int(sys_sc.loc['drug','n_jurisdictions'])} jurisdictions saw a lower all-drug indicator (median ratio {sys_sc.loc['drug','median_ratio_2025_vs_2023']:.2f}), and the corresponding figures were {sys_sc.loc['opioid','share_declined']*100:.0f}% for opioids (median ratio {sys_sc.loc['opioid','median_ratio_2025_vs_2023']:.2f}) and {sys_sc.loc['stimulant','share_declined']*100:.0f}% for stimulants (median ratio {sys_sc.loc['stimulant','median_ratio_2025_vs_2023']:.2f}). The opioid series had its estimated steepest decline around {mon(sys_cp.loc['opioid','bp2_month'])} (95% bootstrap interval {mon(sys_cp.loc['opioid','bp2_boot95_lo'])} to {mon(sys_cp.loc['opioid','bp2_boot95_hi'])}), but the all-drug and stimulant breakpoints lay near the edges of the search range or had very wide intervals and are not interpreted.")
    e18 = dxp[(dxp.dataset == "ED") & (dxp.indicator == "count_alldrug")].set_index("year")
    h18 = dxp[(dxp.dataset == "HOSP") & (dxp.indicator == "count_alldrug")].set_index("year")
    p(f"The discharge data show a similar fall at the end of the period. In {int(e18.n_jurisdictions.iloc[0])} jurisdictions with complete emergency department counts, the crude all-drug rate was {e18.loc[2018,'crude_rate_per_100k']:.1f} per 100,000 in 2018, {e18.loc[2021,'crude_rate_per_100k']:.1f} in 2021 and {e18.loc[2024,'crude_rate_per_100k']:.1f} in 2024; inpatient rates in {int(h18.n_jurisdictions.iloc[0])} jurisdictions went from {h18.loc[2018,'crude_rate_per_100k']:.1f} to {h18.loc[2024,'crude_rate_per_100k']:.1f}. Within jurisdictions, the monthly DOSE-SYS indicator and the discharge emergency department rate agreed in rank (median Spearman correlation {conc.spearman_rho.median():.2f}, range {conc.spearman_rho.min():.2f} to {conc.spearman_rho.max():.2f}, {len(conc)} jurisdictions), although the two measures use different denominators and case definitions.")

    h2("3.6 Early-warning performance")
    pi = ewi.loc[PRIM]
    p(f"The primary analysis included {int(pi.n_jurisdictions_in_fatal_panel)} cohort jurisdictions of which {int(ewm[(ewm.config==PRIM)&(ewm.period=='test')].n_jurisdictions.max())} had usable DOSE-SYS features, giving {int(pi.n_validation_rows_h1_L0)} validation and {int(pi.n_test_rows_h1_L0)} test jurisdiction-months at h = 1. On the validation year the selected augmented model used the {SEL.feature_set} indicator with ridge penalty {SEL.alpha_augmented}; the baseline regression used penalty {SEL.alpha_baseline}. The 2024 test period was a stern test, because deaths fell in every month while the baselines carry forward earlier levels. Persistence and the moving average over-predicted in aggregate and the seasonal-naive forecast, which repeats 2023, was far worse than every alternative (Table 6).")
    t00 = test_prim.loc[(0, 1)]
    t61, t62, t63 = test_prim.loc[(6, 1)], test_prim.loc[(6, 2)], test_prim.loc[(6, 3)]
    t31, t32, t33 = test_prim.loc[(3, 1)], test_prim.loc[(3, 2)], test_prim.loc[(3, 3)]
    p(f"When fatal history was current (L = 0), adding DOSE-SYS features did not improve forecasts: the change in mean Poisson deviance relative to the fatal-history regression was {pct(t00.relative_diff_pct,1,True)} at h = 1 (95% CI {pct(t00.rel_ci_lo_pct,1,True)} to {pct(t00.rel_ci_hi_pct,1,True)}), and all three horizons had intervals that included zero. With fatal data lagged by three months the point estimates favoured the augmented model by {pct(abs(t31.relative_diff_pct),1)}, {pct(abs(t32.relative_diff_pct),1)} and {pct(abs(t33.relative_diff_pct),1)} at h = 1, 2 and 3, and with a six-month lag by {pct(abs(t61.relative_diff_pct),1)}, {pct(abs(t62.relative_diff_pct),1)} and {pct(abs(t63.relative_diff_pct),1)}; only the six-month lag at h = 2 and h = 3 had intervals excluding zero (Figure 5a). Against the simplest baseline the picture is mixed: at a six-month lag the augmented regression was not better than persistence, with differences of {ewd[(ewd.period=='test')&(ewd.config==PRIM)&(ewd.model_b=='persistence')&(ewd.L==6)].relative_diff_pct.min():+.1f}% to {ewd[(ewd.period=='test')&(ewd.config==PRIM)&(ewd.model_b=='persistence')&(ewd.L==6)].relative_diff_pct.max():+.1f}% in mean deviance and intervals spanning zero.")
    vv = val_prim.relative_diff_pct
    p(f"The small gains in 2024 were not foreshadowed in the validation year, where the same comparison ranged from {vv.min():+.1f}% to {vv.max():+.1f}% across the nine lag-by-horizon combinations and included an apparent loss of {val_prim.loc[(3,3)].relative_diff_pct:+.1f}% at L = 3 and h = 3. The pattern is thus not stable between years. Interval coverage was also imperfect: with current fatal history the 80% intervals of the regression models covered about {ewm[(ewm.config==PRIM)&(ewm.period=='test')&(ewm.L==0)&(ewm.model=='glm_aug')].coverage80.mean()*100:.0f}% of observations, and with a six-month lag only about {ewm[(ewm.config==PRIM)&(ewm.period=='test')&(ewm.L==6)&(ewm.model=='glm_aug')].coverage80.mean()*100:.0f}%, reflecting the sustained downward drift.")
    s_all = ewd[(ewd.period == "test") & (ewd.config == SENS_ALL) & (ewd.model_b == "glm_base")].set_index(["L", "h"])
    s_dx = ewd[(ewd.period == "test") & (ewd.config == SENS_DX) & (ewd.model_b == "glm_base")].set_index(["L", "h"])
    p(f"Sensitivity analyses led to the same reading. Using all jurisdictions with usable data ({int(ewi.loc[SENS_ALL].n_test_rows_h1_L0)} test rows at h = 1) the augmented model was slightly worse at L = 0, h = 1 ({pct(s_all.loc[(0,1)].relative_diff_pct,1,True)}, interval {pct(s_all.loc[(0,1)].rel_ci_lo_pct,1,True)} to {pct(s_all.loc[(0,1)].rel_ci_hi_pct,1,True)}) and modestly better at a six-month lag ({pct(s_all.loc[(6,2)].relative_diff_pct,1,True)} at h = 2). Replacing DOSE-SYS by the discharge-based emergency department rate in {int(ewm[(ewm.config==SENS_DX)&(ewm.period=='test')].n_jurisdictions.max())} jurisdictions gave {pct(s_dx.loc[(0,1)].relative_diff_pct,1,True)} at L = 0, h = 1 (worse), and between {pct(s_dx.loc[(6,1)].relative_diff_pct,1,True)} and {pct(s_dx.loc[(3,1)].relative_diff_pct,1,True)} at the longer lags. Discharge data are typically slower than syndromic data, so these estimates describe information content, not operational timeliness.")
    lo = ll[ll.indicator == "opioid"].set_index("lead_months")
    p(f"The lead\u2013lag profile gives a possible reason. Pooled across jurisdiction-months, the correlation between nonfatal and fatal momentum was strongest at zero lead and decayed with lead time: for the opioid indicator rho was {lo.loc[0,'spearman_rho']:.2f} (95% CI {lo.loc[0,'ci_lo']:.2f} to {lo.loc[0,'ci_hi']:.2f}) at zero months, {lo.loc[2,'spearman_rho']:.2f} at two months and {lo.loc[4,'spearman_rho']:.2f} at four months (interval {lo.loc[4,'ci_lo']:.2f} to {lo.loc[4,'ci_hi']:.2f}). The indicators move with mortality, but within this panel they do not appear to move ahead of it by more than a couple of months.")
    B.append({"type": "table", "title": f"Table 6. Out-of-sample performance in the 2024 test year, primary analysis (DOSE-SYS features, {int(ewm[(ewm.config==PRIM)&(ewm.period=='test')].n_jurisdictions.max())} jurisdictions with usable features). Mean Poisson deviance by model, and the percentage change in deviance from adding DOSE-SYS features to the fatal-history regression.", "key": "T6"})
    B.append({"type": "figure", "path": str(FIG / "Figure5_earlywarning_performance.png"), "caption": "Figure 5. (a) Percentage change in mean Poisson deviance when DOSE-SYS momentum features are added to a fatal-history regression, 2024 test year; negative values favour the augmented model; bars are 95% bootstrap intervals over jurisdictions. (b) Aggregate observed deaths and forecasts for a three-month lag and three-month horizon. Source data: Figure5_earlywarning_performance_source_data.csv.", "width": 620})

    h2("3.7 Geographic variation and robustness")
    p(f"The 2023 to 2024 decline occurred in all {int(re_last.k_jurisdictions)} cohort jurisdictions (Figure 6). The random-effects pooled change was {pct(re_last.pooled_pct_change,1,True)} (95% CI {pct(re_last.ci_lo,1,True)} to {pct(re_last.ci_hi,1,True)}), with substantial heterogeneity (I\u00b2 = {re_last.I2_pct:.0f}%) and a 95% prediction interval of {pct(re_last.pred_lo,1,True)} to {pct(re_last.pred_hi,1,True)} for a new jurisdiction. Regionally, the cohort\u2019s western jurisdictions declined least ({pct(reg.loc['West','pct_change'],1,True)}, {int(reg.loc['West','n'])} jurisdictions) against {pct(reg.loc['Northeast','pct_change'],1,True)} in the Northeast, {pct(reg.loc['Midwest','pct_change'],1,True)} in the Midwest and {pct(reg.loc['South','pct_change'],1,True)} in the South. The earlier period was far more variable: between 2020 and 2023 the pooled change was {pct(re_first.pooled_pct_change,1,True)} but only {re_first.share_declined*100:.0f}% of jurisdictions recorded a decline, with a prediction interval from {pct(re_first.pred_lo,1,True)} to {pct(re_first.pred_hi,1,True)} (I\u00b2 = {re_first.I2_pct:.0f}%). Jurisdiction intervals assume Poisson variation and are therefore optimistic.")
    c0 = comp.iloc[0]; c3 = comp.iloc[3]
    others = comp[comp.definition.str.contains("excluding") & ~comp.definition.str.contains("West") & comp.definition.str.contains("Cohort27")].pct_change_2023_2024.sub(c0.pct_change_2023_2024).abs().max()
    p(f"Jurisdiction composition mattered for levels but little for the 2023 to 2024 change in the fixed cohort and paired sets. The change was {pct(c0.pct_change_2023_2024,1,True)} in the 27-jurisdiction cohort and {pct(comp.iloc[1].pct_change_2023_2024,1,True)} in all {int(comp.iloc[1].n_jurisdictions)} jurisdictions present in both years. Taking the dashboard\u2019s composite rows at face value, whose membership changed from 42 to 43 jurisdictions, gave {pct(c3.pct_change_2023_2024,1,True)}, an overstatement of the fall by {abs(c3.pct_change_2023_2024-c0.pct_change_2023_2024):.1f} percentage points that arises from composition alone. Removing the West raised the estimated decline to {pct(comp[comp.definition.str.contains('excluding West')].pct_change_2023_2024.iloc[0],1,True)}, and removing any other region changed it by at most {others:.1f} percentage points (Table 7).")
    sp = cps[cps.series == "alldrug"].set_index("specification").bp_month
    jka = jk[jk.series == "alldrug"].bp_month.value_counts()
    p(f"The all-drug breakpoint was stable across specifications, falling between {mon(sp.min())} and {mon(sp.max())} (Figure 7a); with 2020 removed it was {mon(sp['series starts 2021-01 (excludes 2020)'])}. Leaving out one jurisdiction at a time placed it in {', '.join(f'{mon(k)} ({v} fits)' for k, v in jka.items())} (Figure 7b), so no single jurisdiction drove the result. Truncating the final three or six months did not change the timing materially, although we cannot assess reporting delay directly because the workbooks contain a single data vintage and no provisional-versus-final comparison is possible.")
    B.append({"type": "table", "title": "Table 7. Sensitivity of the 2023 to 2024 change in all-drug deaths to the definition of the jurisdiction set, and of the all-drug breakpoint to specification.", "key": "T7"})
    B.append({"type": "figure", "path": str(FIG / "Figure6_geographic_variation.png"), "caption": "Figure 6. Change in all-drug overdose deaths from 2023 to 2024 in each of the 27 cohort jurisdictions with Poisson 95% intervals, grouped by census region. Source data: Figure6_geographic_variation_source_data.csv.", "width": 430})
    B.append({"type": "figure", "path": str(FIG / "Figure7_robustness.png"), "caption": "Figure 7. (a) Single-break estimates for four series under alternative specifications. (b) Leave-one-jurisdiction-out distribution of the single-break location. Source data: Figure7_robustness_source_data.csv.", "width": 620})

    # ------------------------------------------------------------ 4 Discussion
    h1("4. Discussion")
    p(f"In a fixed set of {int(QC('Jurisdictions included in all 5').result)} jurisdictions, overdose deaths rose through 2020, stayed high until mid-2023 and then fell quickly. The timing of the turn was similar for opioid, fentanyl, stimulant and methamphetamine deaths and robust to model choice, to the removal of 2020 and to the removal of any single jurisdiction. The drop coincided with a fall in the share of deaths involving opioids and with a smaller fall in the nonfatal indicators in DOSE-SYS. We did not test any explanation for the downturn, and the data cannot discriminate among those proposed, such as a disruption of the fentanyl supply [K:vangelov], treatment access or naloxone distribution. That opioid and fentanyl deaths fell fastest is compatible with a supply-related explanation but does not establish one. Detecting a break, in particular, is not evidence about its cause.")
    p(f"The central practical lesson from the drug-category analysis is to read share and count together. Almost every non-opioid category gained share between 2022 and 2024, but for most of them the number of deaths fell. Only a handful of toxicology-detected substances increased on both measures. Our results are consistent in direction with the SUDORS reports of fentanyl\u2013stimulant co-involvement [K:odonnell] and with the rising detection of xylazine [K:kariisa], which in our cohort reached about {scr[scr.category=='xylazine'].share_2024_pct.iloc[0]:.1f}% of deaths in 2024 when measured against all drug overdose deaths. They also show that rankings based on recent share alone would flag drugs, such as parafluorofentanyl, whose share had already begun to recede. The stimulant share we observed (about {A('stimulant',2024,'share_pct'):.0f}% in 2024) is not directly comparable with the 59.0% reported for SUDORS jurisdictions as a whole between January 2021 and June 2024 [K:tanz], because jurisdictions and periods differ, but both indicate that stimulants are now present in most deaths. The rise in bromazolam agrees with reports of a surge in bromazolam-related deaths in San Francisco in 2023 [K:rodda] and of frequent unintended exposure among emergency department overdose patients [K:falise]. At the same time, adulterants that were not variables in the workbook cannot appear in our screen at all; medetomidine, for instance, replaced xylazine as the most common adulterant in the Philadelphia opioid supply in late 2024 [K:huo]. Because detection depends on testing practice, our rising-share list is a list of substances worth watching, not a measure of emergence.")
    p("The early-warning findings are deliberately modest. The two closest studies we found examined small areas and shorter lags within one or two states and reported that nonfatal signals can help identify fatal outbreaks [K:patton] and that concurrent associations are strong while lagged ones weaken [K:skinner]. Our state-level, out-of-sample results are compatible with both. The decline in our pooled nonfatal series is consistent with national syndromic reports that fentanyl-involved visit rates peaked in mid-2023 and then fell [K:pickens]. Nonfatal momentum is correlated with fatal momentum at zero lead, adds little when fatal history is current, and offers a small and unstable advantage when fatal history is stale. The advantage appeared in the test year but not in validation, and at the longest lag the augmented regression did not beat simple persistence. Persistence is hard to beat for months in which the series changes slowly, and in 2024 the series changed in only one direction. We therefore do not regard the DOSE-SYS indicators as validated for operational early warning. Their value for jurisdictions without timely mortality data deserves study with county-level data, vintage-specific mortality counts and a longer test window that contains more than one turning point.")
    p(f"Two methodological findings may help other analysts. Composite rows on dashboards can mislead: the published composite rows overstated the 2023 to 2024 decline by about {abs(comp.iloc[3].pct_change_2023_2024-comp.iloc[0].pct_change_2023_2024):.1f} percentage points relative to a fixed cohort. And sentinel codes can silently corrupt aggregation: unscreened 9999 values turn small counts into implausible totals. Our audit found no arithmetic inconsistency in the files, which supports their use for the descriptive analyses presented here.")

    h1("5. Limitations")
    p("The data are aggregate and ecological; nothing here supports statements about individuals. The cohort contains only jurisdictions that met SUDORS inclusion criteria in every year, and these are not a probability sample of the United States; the regional composition is uneven, with, for example, only six western jurisdictions. Because all five years were required, jurisdictions with more recent entry are excluded, which may favour better-resourced reporting systems. The 2024 figures are the most recent available in the file and may be affected by reporting delay or by incomplete toxicology; we cannot examine this because only one vintage was supplied. The SUDORS case definition and the identification of drugs \u2018involved\u2019 or \u2018detected\u2019 depend on medical examiner practice, and the combination categories overlap. Published age-adjusted rates were not used for cohort-level inference because population denominators for cohort composites are not in the file, and differential-privacy changes in census estimates affect rates after 2021.")
    p("DOSE-SYS gives ratios per 10,000 visits without visit counts, so pooled means are unweighted and sensitive to facility coverage and to the composition of syndromic networks, which vary widely by state. Many drug-specific indicators are suppressed, which restricted the predictive analysis to three indicators. The discharge workbook lacks a dictionary. Our forecasting design assumes that mortality records for earlier months are available at the stated lag, which is a simplification of real reporting; it is a retrospective evaluation on 12 test months, 25 jurisdictions and a series with one dominant direction, and the confidence intervals reflect jurisdiction sampling only. Cluster-robust inference rests on 27 clusters and a few annual time points and should be read as approximate. Segmented models assume a piecewise-linear log trend, and boundary solutions are reported rather than hidden. Finally, we screened many categories and used false discovery control, but the screening remains exploratory, and the literature review was focused rather than systematic.")

    h1("6. Conclusions")
    p(f"Within the fixed 27-jurisdiction SUDORS cohort, overdose deaths peaked in 2023, with the turning point estimated near {mon(bp.bp2_month)}, and fell by about a quarter in 2024. The decline affected every drug class and every jurisdiction in the cohort, though unevenly across regions. Shares of deaths shifted from opioids toward stimulants mainly because opioid-involved deaths fell faster, and only a small group of toxicology-detected substances, including bromazolam, carfentanil, nitazenes and ketamine, rose in both share and number. Nonfatal DOSE-SYS indicators fell over the same period and tracked mortality closely at zero lead, but they did not provide reliable additional predictive information in a chronological test; the small gains seen at long reporting lags were inconsistent between validation and test years. These findings apply to participating jurisdictions, are descriptive or exploratory, and make no claim of causation or national representativeness.")

    h1("Data Availability Statement")
    p("The source data are public CDC dashboard downloads: the SUDORS fatal-overdose workbook, the DOSE-SYS workbook (download dated 28 September 2026 in its filename) and the DOSE discharge-data workbook (5 August 2026 in its filename) [K:sudors,dose]. Stable download URLs were not verified in the preparation of this manuscript, and the exact files analysed are provided unmodified with SHA-256 checksums. The derived analytical datasets, model outputs and figure source data are provided in the accompanying package. Redistribution terms of the source files should be confirmed with CDC.")
    h1("Code Availability Statement")
    p("All analyses are implemented in Python and run from a single entry point (04_Python_Code/main.py). The package includes requirements.txt, recorded software versions, random seeds and an analysis log; running main.py on the supplied raw files regenerates the processed tables, figures and automated checks.")
    h1("Ethics Statement")
    p("This is a secondary analysis of publicly available, aggregated, de-identified surveillance data that contain no individual-level records. Such work is generally not considered human-subjects research, but the determination belongs to the authors\u2019 institution and has not been obtained for this manuscript. No ethics approval is claimed.")
    h1("Author Contributions, Funding, Conflicts of Interest and Acknowledgments")
    p("[Author contributions to be completed by the authors.] [Funding statement to be completed.] [Conflict-of-interest declaration to be completed.] [Acknowledgments to be completed.]")
    h1("References")
    ordered = number_citations(B)
    for i, r in enumerate(ordered, 1):
        B.append({"type": "ref", "text": f"[{i}] {r}"})
    return B


def tables():
    T = {}
    d = inv.groupby("file").agg(sheets=("sheet", "count"), size_kb=("file_size_bytes", lambda s: s.iloc[0] / 1024), sha256=("sha256", "first")).reset_index()
    t1 = pd.DataFrame([
        ["SUDORS dashboard workbook", "Fatal overdose; jurisdiction-year with monthly counts, combinations, detected drugs", "2020\u20132024", "45 jurisdictions ever; 27 in all years", "Aggregate counts, percentages, age-adjusted rates"],
        ["DOSE-SYS workbook (09-28-2026)", "Suspected nonfatal ED overdoses per 10,000 ED visits, jurisdiction-month", "Jan 2019\u2013Jul 2026 (Mar\u2013Aug 2020 excluded)", "51 listed; 48 with any numeric value after 2021", "Aggregate ratios; no visit counts"],
        ["DOSE-dx workbook (08-05-2026)", "Nonfatal ED and inpatient discharge counts and crude rates per 100,000", "2018\u20132024", "29\u201330 ED, 34 inpatient jurisdictions", "Aggregate counts; no data dictionary"],
    ], columns=["Dataset", "Content and unit", "Period", "Jurisdictions", "Level of observation"])
    T["T1"] = t1
    rows = []
    for key, lab in [("Duplicate (jurisdiction, data_year)", "SUDORS: duplicate jurisdiction-year keys"), ("Sum of 12 monthly counts equals annual total (alldrug)", "SUDORS: monthly sums equal annual totals (all drugs; all 10 classes passed)"),
                     ("Own 27-jurisdiction cohort totals", "SUDORS: recomputed cohort totals equal published totals"), ("Cells equal to 9999", "SUDORS: cells equal to 9999 (treated as missing)"),
                     ("Published xylazine_percent", "SUDORS: own xylazine share vs published percent (within 0.06 pp)"), ("Duplicate (jurisdiction, year, month, time_unit)", "DOSE-SYS: duplicate keys"),
                     ("Numeric all-drug values in Mar-Aug 2020", "DOSE-SYS: numeric values, Mar\u2013Aug 2020"), ("(drug_proportion)", "DOSE-SYS: numeric share of monthly cells, all-drug"),
                     ("(fentanyl_proportion)", "DOSE-SYS: numeric share of monthly cells, fentanyl"), ("(methamphetamine_proportion)", "DOSE-SYS: numeric share of monthly cells, methamphetamine"),
                     ("Duplicate (jurisdiction, month, year, dataset", "DOSE-dx: duplicate keys"), ("Sum of monthly all-drug ED counts", "DOSE-dx: monthly counts sum to annual"),
                     ("count_fentanyl: numeric", "DOSE-dx: fentanyl counts numeric / suppressed / empty")]:
        r = QC(key)
        rows.append([lab, str(r.result)])
    T["T2"] = pd.DataFrame(rows, columns=["Check", "Result"])
    a = ann.copy()
    rows = []
    for dname in SUDORS_DRUGS:
        row = [SUDORS_DRUGS[dname]]
        for y in [2020, 2021, 2022, 2023, 2024]:
            r = a[(a.drug == dname) & (a.year == y)].iloc[0]
            row.append(f"{int(r.deaths):,}" + ("" if dname == "alldrug" else f" ({r.share_pct:.1f}%)"))
        e = C(dname, 2023, 2024)
        row.append(f"{e[0]:+.1f} ({e[1]:+.1f} to {e[2]:+.1f})".replace("-", "\u2212"))
        rows.append(row)
    T["T3"] = pd.DataFrame(rows, columns=["Drug class", "2020", "2021", "2022", "2023", "2024", "Change 2023\u20132024, % (95% CI)"])
    t4 = rising_up.sort_values("count_change_2023_2024_pct", ascending=False)
    rows = []
    for r in t4.itertuples():
        rows.append([r.category.replace("_", " "), r.family.split(" ")[0], int(r.n_jurisdictions), f"{r.share_2022_pct:.2f}", f"{r.share_2024_pct:.2f}", f"{r.deaths_2023:,} \u2192 {r.deaths_2024:,}", f"{r.count_change_2023_2024_pct:+.1f}", f"{r.recent_or_per_year:.2f} ({r.recent_or_lo:.2f}\u2013{r.recent_or_hi:.2f})", fmtq(r.recent_q_bh)])
    T["T4"] = pd.DataFrame(rows, columns=["Category", "Type", "Jur.", "Share 2022, %", "Share 2024, %", "Deaths 2023 \u2192 2024", "Change in deaths, %", "Odds ratio per year (95% CI)", "BH q"])
    rows = []
    lab = {"alldrug": "All drugs", "opioids": "Any opioid", "imfs": "Illicit fentanyls", "heroin": "Heroin", "rxopioids": "Rx opioids", "stimulant": "Stimulants", "cocaine": "Cocaine", "meth": "Methamphetamine", "nonopioid_sedatives": "Non-opioid sedatives", "benzodiazepines": "Benzodiazepines"}
    for s, r in cp.iterrows():
        def b(i):
            m = r.get(f"bp{i}_month")
            if pd.isna(m):
                return "\u2014"
            flag = " (boundary)" if bool(r.get(f"bp{i}_at_boundary")) else ""
            return f"{m} [{r.get(f'bp{i}_boot95_lo')}, {r.get(f'bp{i}_boot95_hi')}]{flag}"
        sl = " / ".join(f"{r[f'segment{i}_annual_pct']:+.0f}" for i in (1, 2, 3) if f"segment{i}_annual_pct" in r and not pd.isna(r[f"segment{i}_annual_pct"]))
        rows.append([lab[s], int(r.selected_breakpoints), b(1), b(2), sl])
    T["T5"] = pd.DataFrame(rows, columns=["Series", "Breaks (BIC)", "Break 1 [95% bootstrap]", "Break 2 [95% bootstrap]", "Segment slopes, % per year"])
    rows = []
    mt = ewm[(ewm.config == PRIM) & (ewm.period == "test")]
    for (L, h), g in mt.groupby(["L", "h"]):
        g = g.set_index("model")
        dd = test_prim.loc[(L, h)]
        rows.append([L, h, int(g.n_rows.iloc[0]), f"{g.loc['persistence','mean_poisson_deviance']:.2f}", f"{g.loc['ma3','mean_poisson_deviance']:.2f}", f"{g.loc['snaive','mean_poisson_deviance']:.2f}",
                     f"{g.loc['glm_base','mean_poisson_deviance']:.2f}", f"{g.loc['glm_aug','mean_poisson_deviance']:.2f}",
                     f"{dd.relative_diff_pct:+.1f} ({dd.rel_ci_lo_pct:+.1f} to {dd.rel_ci_hi_pct:+.1f})".replace("-", "\u2212"), f"{g.loc['glm_aug','mae']:.1f} / {g.loc['glm_base','mae']:.1f}"])
    T["T6"] = pd.DataFrame(rows, columns=["Lag L", "Horizon h", "Rows", "Persist.", "3-mo avg", "Seasonal naive", "GLM fatal hist.", "GLM + DOSE-SYS", "Change from adding DOSE-SYS, % (95% CI)", "MAE aug / base"])
    rows = [[r.definition, "" if pd.isna(r.n_jurisdictions) else int(r.n_jurisdictions), f"{int(r.deaths_2023):,}", f"{int(r.deaths_2024):,}", f"{r.pct_change_2023_2024:+.1f}".replace("-", "\u2212")] for r in comp.itertuples()]
    t7a = pd.DataFrame(rows, columns=["Jurisdiction set", "Jur.", "Deaths 2023", "Deaths 2024", "Change, %"])
    sp = cps[cps.series == "alldrug"][["specification", "bp_month"]]
    for r in sp.itertuples():
        t7a.loc[len(t7a)] = [f"All-drug breakpoint: {r.specification}", "", "", "", r.bp_month]
    T["T7"] = t7a
    for k, v in T.items():
        v.to_csv(PROC / f"manuscript_table_{k}.csv", index=False)
    return T


if __name__ == "__main__":
    _t = test_prim.reset_index()
    excl = {(int(r.L), int(r.h)) for r in _t.itertuples() if r.ci_hi < 0 or r.ci_lo > 0}
    assert excl == {(6, 2), (6, 3)}, f"Text claim about intervals excluding zero no longer matches data: {excl}"
    assert not any(r.ci_hi < 0 or r.ci_lo > 0 for r in _t[_t.L == 0].itertuples()), "L=0 claim violated"
    _p = ewd[(ewd.period == "test") & (ewd.config == PRIM) & (ewd.model_b == "persistence") & (ewd.L == 6)]
    assert (_p.rel_ci_lo_pct < 0).all() and (_p.rel_ci_hi_pct > 0).all(), "persistence claim violated"
    blocks = build()
    T = tables()
    for b in blocks:
        if b["type"] == "table":
            df = T[b["key"]]
            b["header"] = list(df.columns)
            b["rows"] = df.astype(str).values.tolist()
    # order tables numerically by moving them is unnecessary: they are placed where first cited
    # citation integrity is enforced inside number_citations (unknown or unused keys abort the build)
    n_refs = sum(1 for b in blocks if b["type"] == "ref")
    assert n_refs >= 20, f"only {n_refs} references"
    (PROC / "manuscript_spec.json").write_text(json.dumps(blocks, ensure_ascii=False))
    print("blocks", len(blocks), "tables", list(T), "references", n_refs, "all cited")
