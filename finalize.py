"""Documentation, supplementary workbook, provenance manifest, checksums and manuscript-number verification.

Run after main.py, manuscript_content.py and make_docx.js (see run_all.sh)."""
import hashlib
import subprocess
from datetime import datetime

import pandas as pd

from config import DOC, FIG, PROC, RAW, ROOT, SUPP

P = lambda f: pd.read_csv(PROC / f)  # noqa: E731


def supplementary():
    sheets = {
        "S1 trend models": "fatal_trend_models.csv", "S2 category screening": "fatal_category_screening.csv",
        "S3 changepoints": "fatal_changepoints.csv", "S4 changepoint sensitivity": "fatal_changepoint_sensitivity.csv",
        "S5 jackknife": "fatal_changepoint_jackknife.csv", "S6 cohort changes": "fatal_cohort27_changes.csv",
        "S7 SYS coverage": "dose_sys_indicator_coverage.csv", "S8 SYS cohorts": "dose_sys_cohorts.csv",
        "S9 SYS changepoints": "dose_sys_changepoints.csv", "S10 dx pooled": "dose_dx_pooled_annual.csv",
        "S11 EW metrics": "earlywarning_met.csv", "S12 EW differences": "earlywarning_diff.csv",
        "S13 EW selection": "earlywarning_sel.csv", "S14 EW baseline sel": "earlywarning_base.csv",
        "S15 EW lead-lag": "earlywarning_lead_lag.csv", "S16 geo state": "geo_state_changes.csv",
        "S17 geo random eff": "geo_random_effects.csv", "S18 composition": "robust_composition.csv",
        "S19 filter log": "filter_log.csv", "S20 QC checks": "qc_checks.csv", "S21 automated checks": "automated_checks.csv",
    }
    with pd.ExcelWriter(SUPP / "Supplementary_Tables.xlsx") as xw:
        for name, f in sheets.items():
            P(f).to_excel(xw, sheet_name=name[:31], index=False)
    pd.read_csv(PROC / "fatal_changepoints.csv").to_csv(SUPP / "changepoints_full.csv", index=False)


PROVENANCE = [
    ("Table 1", "Dataset characteristics", "data_inventory.csv (07_Documentation); manuscript_table_T1.csv", "ingest.build_inventory; manuscript_content.tables"),
    ("Table 2", "Data quality", "qc_checks.csv; dose_sys_indicator_coverage.csv", "ingest.qc_checks"),
    ("Table 3 / Fig 2a, 3", "Cohort annual deaths, shares, 2023-24 change", "fatal_cohort27_annual.csv; fatal_cohort27_changes.csv", "analysis_fatal.cohort_annual"),
    ("Text 3.2", "Single-slope trend diagnostics", "fatal_trend_models.csv", "analysis_fatal.trend_models"),
    ("Table 4 / Fig 4", "Change points", "fatal_changepoints.csv", "analysis_fatal.changepoints_fatal"),
    ("Table 5", "Category screening (BH)", "fatal_category_screening.csv", "analysis_fatal.screen_categories"),
    ("Fig 2b / Text 3.5", "DOSE-SYS pooled series and changes", "dose_sys_pooled_monthly.csv; dose_sys_pooled_annual_means.csv; dose_sys_state_changes.csv; dose_sys_changepoints.csv", "analysis_nonfatal"),
    ("Text 3.5", "DOSE-dx pooled rates, concordance", "dose_dx_pooled_annual.csv; dose_sys_dx_concordance.csv", "analysis_nonfatal.dx_pooled / dx_sys_concordance"),
    ("Table 6 / Fig 5", "Early-warning performance", "earlywarning_met.csv; earlywarning_diff.csv; earlywarning_pred.csv", "earlywarning.run_earlywarning"),
    ("Text 3.6", "Lead-lag correlations", "earlywarning_lead_lag.csv", "earlywarning.lead_lag"),
    ("Fig 6 / Text 3.7", "Geographic heterogeneity", "geo_state_changes.csv; geo_random_effects.csv; geo_region_summary.csv", "robustness"),
    ("Table 7 / Fig 7", "Composition and breakpoint sensitivity", "robust_composition.csv; fatal_changepoint_sensitivity.csv; fatal_changepoint_jackknife.csv", "robustness.composition_sensitivity; analysis_fatal.changepoint_sensitivity"),
]


def provenance():
    pd.DataFrame(PROVENANCE, columns=["manuscript_item", "content", "source_files (03_Processed_Data unless noted)", "generating_function"]).to_csv(DOC / "provenance_manifest.csv", index=False)


def sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def checksums():
    lines = []
    for folder in ["02_Raw_Data", "03_Processed_Data", "04_Python_Code", "05_Figures", "06_Supplementary_Materials", "07_Documentation"]:
        for f in sorted((ROOT / folder).rglob("*")):
            if f.is_file() and f.name != "sha256_manifest.txt" and "__pycache__" not in str(f):
                lines.append(f"{sha(f)}  {f.relative_to(ROOT)}")
    for f in ["01_Research_Manuscript.docx", "01_Research_Manuscript.pdf"]:
        if (ROOT / f).exists():
            lines.append(f"{sha(ROOT / f)}  {f}")
    (DOC / "sha256_manifest.txt").write_text("\n".join(lines) + "\n")
    return len(lines)


def manuscript_numbers_check():
    """Confirm that key computed values appear in the manuscript text (guards against stale builds)."""
    txt = subprocess.run(["pandoc", str(ROOT / "01_Research_Manuscript.docx"), "-t", "plain", "--wrap=none"], capture_output=True, text=True).stdout
    txt = txt.replace("\u2212", "-")
    ann = P("fatal_cohort27_annual.csv")
    tests = []
    for y in range(2020, 2025):
        v = int(ann[(ann.drug == "alldrug") & (ann.year == y)].deaths.iloc[0])
        tests.append((f"all-drug deaths {y}", f"{v:,}"))
    cp = P("fatal_changepoints.csv").set_index("series")
    tests.append(("all-drug break 2", pd.Timestamp(cp.loc["alldrug"].bp2_month + "-01").strftime("%B %Y")))
    ch = P("fatal_cohort27_changes.csv")
    r = ch[(ch.drug == "alldrug") & (ch.from_year == 2023) & (ch.to_year == 2024)].iloc[0]
    tests.append(("2023-24 change", f"{r.relative_change_pct:+.1f}%".replace("+", "")))
    rows = [{"check": n, "expected_string": s, "found_in_manuscript": s in txt} for n, s in tests]
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "manuscript_number_checks.csv", index=False)
    bad = out[~out.found_in_manuscript]
    if len(bad):
        raise SystemExit(f"Manuscript numbers not found: {bad}")
    # placeholder scan: only the declared author placeholders may remain
    import re
    ph = re.findall(r"\[[^\]]*(?:to be completed|TODO|TBD|XX)[^\]]*\]", txt, flags=re.I)
    (PROC / "manuscript_placeholders.txt").write_text("\n".join(ph))
    return out, ph


def docs():
    qc = P("qc_checks.csv")
    inv = pd.read_csv(DOC / "data_inventory.csv")
    flog = P("filter_log.csv")
    ac = P("automated_checks.csv")
    (DOC / "data_quality_report.md").write_text(
        "# Data-quality report\n\nGenerated by `ingest.qc_checks` and `harmonize.py`. Raw files were never modified.\n\n"
        "## Key findings\n\n"
        "- All three workbooks are **aggregate** (no individual-level records). SUDORS: jurisdiction-year with monthly counts; DOSE-SYS: jurisdiction-month ratios per 10,000 ED visits (no visit counts); DOSE-dx: jurisdiction-month counts and crude rates.\n"
        "- SUDORS composite 'Overall' rows change membership by year (e.g. 42 jurisdictions in 2023 and 43 in 2024) and are **not comparable across years**; a fixed cohort of 27 jurisdictions was used. Recomputed cohort totals equal the published trend-tab totals.\n"
        "- SUDORS sentinel 9999 appears in 38,084 cells (none in monthly death counts) and was treated as missing. Naive summation of such cells produces implausible totals.\n"
        f"- Each pairwise combination is stored twice in SUDORS; {int(flog[flog.step.str.contains('mirror duplicates')].rows_before.iloc[0])} combination variables reduce to {int(flog[flog.step.str.contains('mirror duplicates')].rows_after.iloc[0])} after name-based de-duplication (value equality alone would wrongly merge all-9999 columns).\n"
        "- Published 'detected drug' percentages use a toxicology-known denominator absent from the file; own shares (deaths / all drug deaths) are conservative.\n"
        "- DOSE-SYS: no values for March-August 2020 (excluded by CDC); fentanyl, heroin, cocaine, methamphetamine and benzodiazepine indicators are mostly suppressed or unreported; only all-drug, opioid and stimulant indicators have usable cohorts.\n"
        "- DOSE-dx has no data dictionary; the meaning of empty (NaN) count cells is undocumented and they were treated as unavailable, never as zero.\n"
        "- No preliminary-versus-final vintages are available; reporting delay therefore cannot be measured directly.\n\n"
        "## All checks\n\n" + qc.to_markdown(index=False) + "\n\n## Filtering log (rows before/after)\n\n" + flog.to_markdown(index=False) + "\n\n## Automated numerical checks\n\n" + ac.to_markdown(index=False) + "\n")
    (DOC / "analyses_not_completed.md").write_text(
        "# Analyses not completed or not supported by the data\n\n"
        "| Analysis | Reason |\n|---|---|\n"
        "| Preliminary vs final observations (RQ5) | Only one data vintage per source was supplied. |\n"
        "| Reporting-delay estimation | No report-date or vintage fields; delay was represented by scenario lags L = 0, 3, 6 months. |\n"
        "| Population-adjusted cohort death rates | Published rates are age-adjusted per jurisdiction; cohort population denominators are not in the SUDORS file. Counts and shares used instead. |\n"
        "| Case-level co-occurrence networks | Data are aggregate; no individual records exist. Only published pairwise combination counts were screened. |\n"
        "| Drug-specific predictive models (fentanyl, heroin, cocaine, methamphetamine, benzodiazepine) | DOSE-SYS indicators for these drugs are suppressed/unreported for most jurisdiction-months (3-10 complete jurisdictions). |\n"
        "| 41 of 108 SUDORS categories in the share screening | Fewer than 10 jurisdictions with valid counts in all five years, or fewer than 100 deaths in 2024 (mostly nitazene/MDMA/cathinone combinations). |\n"
        "| DOSE race/ethnicity, sex/age and county sheets | Not needed for the research questions; not analysed (documented in the inventory). |\n"
        "| Seasonality estimation beyond one harmonic | Only 60 monthly fatal observations per series. |\n"
        "| Verification of JCR quartiles | JCR is not directly accessible from this environment; see journal_shortlist.md. |\n")
    (DOC / "journal_shortlist.md").write_text(
        "# Target-journal shortlist (verification status stated for each item)\n\n"
        "No journal was assumed to be Q1 on the basis of its own claims. Web of Science Journal Citation Reports (JCR) could not be queried directly; the only quartile evidence retrieved is from a third-party aggregator and must be checked by the authors in JCR for the current year before submission.\n\n"
        "| Journal | Fit | Quartile evidence retrieved | Status |\n|---|---|---|---|\n"
        "| Drug and Alcohol Dependence (Elsevier) | Epidemiology and methods in drug use; scope explicitly includes epidemiological studies | Aggregator page (Peeref) lists JCR 2024 categories Psychiatry Q1 and Substance Abuse Q1 | Secondary source; verify in JCR |\n"
        "| Addiction (Wiley) | Published the closest nonfatal-to-fatal early-warning study found (Patton et al., 2025) | Not retrieved | Unverified |\n"
        "| American Journal of Epidemiology (OUP) | Published the Rhode Island nonfatal-to-fatal association study (Skinner et al., 2026) | Not retrieved | Unverified |\n\n"
        "Author guidelines (word limits, structure, reference style) were **not** retrieved; the manuscript uses a conventional IMRaD structure with numbered references and should be reformatted after the target is chosen. The most natural fit for the completed study (descriptive/exploratory surveillance analysis with a negative-to-modest forecasting result) is Drug and Alcohol Dependence, subject to the authors' check of its current article types.\n")
    (DOC / "README.md").write_text(
        "# Overdose surveillance early-warning study - reproducibility package\n\n"
        "## Contents\n\n- `01_Research_Manuscript.docx` (+ PDF rendering)\n- `02_Raw_Data/` original CDC workbooks (unmodified; SHA-256 in `07_Documentation/raw_data_sha256.txt`)\n"
        "- `03_Processed_Data/` cleaned tables, model outputs, forecasts, metrics, table sources\n- `04_Python_Code/` pipeline\n- `05_Figures/` PNG, PDF and source-data CSV for every figure\n"
        "- `06_Supplementary_Materials/` `Supplementary_Tables.xlsx`\n- `07_Documentation/` inventory, dictionary, QC, logs, provenance, limitations\n\n"
        "## Reproduce\n\n```bash\ncd 04_Python_Code\npip install -r requirements.txt\nbash run_all.sh     # main.py -> manuscript_content.py -> make_docx.js -> finalize.py\n```\n"
        "`make_docx.js` needs Node.js with the `docx` package; `finalize.py` uses pandoc to check manuscript numbers. Runtime is about 4-5 minutes. Random seed 20261020.\n\n"
        "## Pipeline\n\n1. `ingest.py` inventory, dictionary, QC checks  2. `harmonize.py` processed tables  3. `analysis_fatal.py` trends, screening, change points  "
        "4. `analysis_nonfatal.py` DOSE-SYS / DOSE-dx  5. `earlywarning.py` rolling-origin forecasting  6. `robustness.py` geography and composition  7. `figures.py`  8. `main.py` orchestration and automated checks.\n\n"
        "## Notes\n\n- The manuscript text is generated from result CSVs; author names, funding, conflicts and acknowledgments are placeholders to complete.\n"
        "- Reference verification status is in `07_Documentation/final_validation_report.md`.\n")
    (ROOT / "04_Python_Code" / "run_all.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 main.py\npython3 manuscript_content.py\nNODE_PATH=$(npm root -g) node make_docx.js\npython3 finalize.py\n")


if __name__ == "__main__":
    supplementary()
    provenance()
    docs()
    out, ph = manuscript_numbers_check()
    n = checksums()
    print("manuscript number checks:", out.found_in_manuscript.all(), "| placeholders:", ph, "| checksummed files:", n)
