"""Central configuration for the overdose early-warning pipeline.

All paths are relative to the project root so the pipeline can be re-run from
any location. Nothing in this file depends on analysis results.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_Raw_Data"
PROC = ROOT / "03_Processed_Data"
FIG = ROOT / "05_Figures"
SUPP = ROOT / "06_Supplementary_Materials"
DOC = ROOT / "07_Documentation"

FILES = {
    "sudors": "SUDORS-Fatal-Overdose-Data.xlsx",
    "sys": "DOSE_SYS_Download_09-28-2026.xlsx",
    "dx": "DOSE_dx_Dashboard_08052026.xlsx",
}

SEED = 20261020            # master seed (analysis date)
N_BOOT = 2000              # bootstrap replicates for state-cluster resampling

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# SUDORS drug classes available with monthly counts (column prefix -> label)
SUDORS_DRUGS = {
    "alldrug": "All drugs",
    "opioids": "Any opioid",
    "imfs": "Illicitly manufactured fentanyls",
    "heroin": "Heroin",
    "rxopioids": "Prescription opioids",
    "stimulant": "Any stimulant",
    "cocaine": "Cocaine",
    "meth": "Methamphetamine",
    "nonopioid_sedatives": "Non-opioid sedatives",
    "benzodiazepines": "Benzodiazepines",
}

# "Drugs of interest detected" families (postmortem toxicology; annual only)
DETECTED = ["carfentanil", "parafluorofentanyl", "other_illegal_opioids", "nitazenes",
            "isotonitazene", "metonitazene", "adulterants_sedatives", "xylazine",
            "gabapentin", "illegal_benzodiazepines", "bromazolam", "etizolam",
            "flualprazolam", "ketamine", "kratom", "mdma_mda",
            "synthetic_cannabinoid", "cathinones"]

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# U.S. Census Bureau regions (used only for descriptive stratification)
_REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
REGION = {st: r for r, s in _REGIONS.items() for st in s.split()}

# Calendar window with no usable DOSE-SYS monthly values (documented by CDC)
SYS_GAP = ("2020-03-01", "2020-08-01")
