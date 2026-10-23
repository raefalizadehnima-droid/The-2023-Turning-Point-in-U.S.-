# Final validation report

**Execution status:** the full pipeline (`main.py` -> `manuscript_content.py` -> `make_docx.js` -> `finalize.py`) ran end to end without error on the supplied raw files. Runtime about 3-5 minutes.

## Completed and verified
- Raw files preserved unmodified with SHA-256 (`raw_data_sha256.txt`, `sha256_manifest.txt`).
- 42 data-quality checks passed; 11 automated numerical checks passed (`03_Processed_Data/automated_checks.csv`), including agreement with CDC's published first-vs-last-year changes for the 27-jurisdiction cohort.
- Manuscript numbers spot-checked against result CSVs (`manuscript_number_checks.csv`); qualitative claims about which forecast intervals exclude zero are asserted against the data at build time; every reference is cited in the text.
- Word file validated against the OOXML schema and rendered to PDF (18 pages); figures and tables inspected visually.
- Seven figures (PNG, PDF, source CSV) and seven tables generated from the same result files.

## Known limitations and unresolved items
1. **Reference verification (23 references; 17 dated 2023-2026).** Checked against publisher, PubMed Central, CDC, or institutional repository pages in this session: Kariisa 2023, O'Donnell 2021 (first author only listed), Patton 2025, Skinner 2026, Garnett 2024 (NCHS Data Brief 522), Spencer 2023 (NCHS Data Brief 474), Vangelov 2026 (Science), Pickens 2025 (MMWR), Khezri 2026 (Int J Drug Policy), Huo 2025 (MMWR), Tanz 2025 (MMWR; DOI follows the MMWR URL pattern mm7432a1 and volume/issue were read from the CDC page, but page numbers were not retrieved), Cano 2024 (JAMA Netw Open), Rodda 2024 (Addiction; start page only, DOI not retrieved), Falise 2025 (Drug Alcohol Depend; volume/pages not retrieved), Mortman 2025 (Pain Med), Bauer 2023 (JMIR Public Health Surveill). Please complete the missing volume/page/DOI fields for Tanz, Rodda and Falise from the publishers. **Not web-verified (cited from knowledge, please check):** Killick 2012, Benjamini & Hochberg 1995, Muggeo 2003, DerSimonian & Laird 1986, Wilson 1927. Dataset references name the downloaded files; stable CDC URLs were not verified. Every in-text citation was assigned by key and the build aborts on any unused or unknown reference.
2. **Literature review** was focused, not systematic; novelty is claimed only narrowly ("not identified in our focused search").
3. **Journal quartiles** could not be verified in JCR (`journal_shortlist.md`); author guidelines were not retrieved.
4. **Placeholders** remain only for author names, contributions, funding, conflicts, acknowledgments, and the institutional ethics determination.
5. **Data limits:** single data vintage (no provisional-vs-final analysis); SUDORS download date unknown; DOSE-dx lacks a dictionary; see `analyses_not_completed.md`.
6. **Interpretation limits:** exploratory analyses; participating jurisdictions only; the early-warning result is small, inconsistent between validation and test years, and not operationally validated.
