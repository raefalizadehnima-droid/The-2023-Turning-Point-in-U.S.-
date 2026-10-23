# Overdose surveillance early-warning study - reproducibility package

## Contents

- `01_Research_Manuscript.docx` (+ PDF rendering)
- `02_Raw_Data/` original CDC workbooks (unmodified; SHA-256 in `07_Documentation/raw_data_sha256.txt`)
- `03_Processed_Data/` cleaned tables, model outputs, forecasts, metrics, table sources
- `04_Python_Code/` pipeline
- `05_Figures/` PNG, PDF and source-data CSV for every figure
- `06_Supplementary_Materials/` `Supplementary_Tables.xlsx`
- `07_Documentation/` inventory, dictionary, QC, logs, provenance, limitations

## Reproduce

```bash
cd 04_Python_Code
pip install -r requirements.txt
bash run_all.sh     # main.py -> manuscript_content.py -> make_docx.js -> finalize.py
```
`make_docx.js` needs Node.js with the `docx` package; `finalize.py` uses pandoc to check manuscript numbers. Runtime is about 4-5 minutes. Random seed 20261020.

## Pipeline

1. `ingest.py` inventory, dictionary, QC checks  2. `harmonize.py` processed tables  3. `analysis_fatal.py` trends, screening, change points  4. `analysis_nonfatal.py` DOSE-SYS / DOSE-dx  5. `earlywarning.py` rolling-origin forecasting  6. `robustness.py` geography and composition  7. `figures.py`  8. `main.py` orchestration and automated checks.

## Notes

- The manuscript text is generated from result CSVs; author names, funding, conflicts and acknowledgments are placeholders to complete.
- Reference verification status is in `07_Documentation/final_validation_report.md`.
