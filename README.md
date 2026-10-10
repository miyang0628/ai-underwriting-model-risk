# Managing the Model Risk of AI-Based Insurance Underwriting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-lightgrey.svg)](#)

> **An auditable counterfactual framework linking interpretability, regulatory
> compliance, and loss-ratio impact for AI-based insurance underwriting.**
>
> *Anonymous authors — manuscript under review at an anonymous journal.*

This repository contains the full, reproducible analysis code for an end-to-end
**XGBoost → DiCE → LLM** pipeline for diabetes risk stratification in insurance
underwriting, framed as a problem of **financial-risk governance** (model risk,
interpretability risk, and regulatory-compliance risk). It is applied to
sex-stratified data from the Korea National Health and Nutrition Examination
Survey (KNHANES 2020–2024; *n* = 13,437).

```
KNHANES 2020-2024 (raw SAS)
  -> 01 preprocessing            sex-stratified, complete-case (n=13,437)
  -> 02 EDA                      prevalence (M 30.0% / F 20.0%), descriptives
  -> 03 training + evaluation    7-algorithm Optuna comparison, Platt calibration,
                                 DeLong tests, bootstrap AUC CIs, DCA
  -> 04 lift + cost-benefit      top-40% high-risk; scenario + Monte Carlo PSA
  -> 05 DiCE counterfactuals     individual algorithmic recourse
  -> 06 robustness               inter-model consistency (Jaccard)
  -> 07 SHAP vs DiCE             attribution vs recourse comparison
  -> 08 LLM guide generation     GPT-4o-mini consultation guides        [API]
  -> 09 LLM-as-Judge + crossmodel 3 personas x 7 dims; GPT-4o replication [API]
```

## Repository structure

```
ai-underwriting-model-risk/
├── data/
│   ├── raw/           # KNHANES SAS files (NOT redistributed — see Data access)
│   └── processed/     # intermediate CSV/JSON/model artifacts (generated)
├── notebooks/         # 01–09, each a single self-contained cell
├── results/
│   ├── figures/       # grayscale, dpi 600, PNG + PDF, no in-image captions
│   └── tables/        # result tables (CSV)
├── requirements.txt
├── .env.example       # template for the OpenAI API key (notebooks 08–09)
├── .gitignore
└── LICENSE
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the LLM notebooks (08–09), copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# edit .env:
#   OPENAI_API_KEY=sk-...
#   LLM_MODEL=gpt-4o-mini
```

`.env` is git-ignored and must never be committed.

## Data access

The KNHANES microdata are **not** included (licensing). Download
`hn20_all.sas7bdat … hn24_all.sas7bdat` from the Korea Disease Control and
Prevention Agency (<https://knhanes.kdca.go.kr>) and place them in `data/raw/`.

## Running

Run the notebooks in order (01 → 09). Each notebook is a single cell that
resolves the repository root automatically, reads its inputs from
`data/processed/`, and writes figures/tables to `results/`. Notebooks 08–09
require a valid API key and are the only ones that make network calls.

## Reproducibility

- All stochastic steps are seeded (`SEED = 42`).
- Notebook 03 exposes `N_TRIALS` (Optuna trials; default **100** as in the
  paper) via an environment variable, and `ALGOS` to optionally restrict the
  algorithm set for quick runs.
- Notebooks 05/06 expose `MAX_CASES` / `N_CASES` to cap the number of
  individuals for quick runs (defaults use the full cohort / paper settings).
- Figures are grayscale (seaborn), 600 dpi, saved as both PNG and PDF, with no
  in-image titles or captions (captions belong in the manuscript).
- Package versions are pinned in `requirements.txt`. Minor numeric differences
  from the manuscript can arise from library versions and the stochastic
  hyperparameter search; the pipeline and conclusions are unaffected.

## Key outputs

| Notebook | Figures (`results/figures/`) | Tables (`results/tables/`) |
|---|---|---|
| 02 | `eda_prevalence_by_sex`, `eda_age_distribution` | `prevalence_by_sex`, `descriptive_stats` |
| 03 | `roc_{male,female}`, `calibration_{male,female}`, `decision_curve_male_female` | `performance_{male,female}`, `delong_{male,female}`, `calibration`, `final_model_summary` |
| 04 | `lift_curves_male_female`, `costbenefit_montecarlo`, `costbenefit_tornado` | `lift_summary`, `cost_benefit_scenarios`, `cost_benefit_montecarlo` |
| 05 | `counterfactual_scatter`, `counterfactual_reduction` | `counterfactual_summary` |
| 06 | `robustness_heatmap` | `robustness_consistency` |
| 07 | `shap_importance_{male,female}` | `shap_vs_dice`, `shap_importance_{male,female}` |
| 09 | `judge_scores_heatmap`, `cross_model_agreement` | `judge_scores_by_persona_dimension`, `cross_model_correlation` |


> **Note on example outputs.** `results/` ships with example figures and tables from a representative run. `robustness_heatmap` (notebook 06) requires the LightGBM/CatBoost pipelines, so run notebook 03 with its default multi-algorithm setting before notebook 06. DiCE (05/06) uses the genetic method; `random` is incompatible with pandas 3.x.

> **Conceptual figures.** The manuscript's Figure 1 (end-to-end pipeline) and
> Figure 2 (three-risk → control map) are schematic design assets, not generated
> by this code; they live with the manuscript (LaTeX) sources. All other figures
> are produced by the notebooks above.

## License

Released under the MIT License (see `LICENSE`). KNHANES data are governed by the
KDCA open-data licence and are not redistributed here.
