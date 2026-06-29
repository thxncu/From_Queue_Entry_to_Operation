# Reproduction: stage-specific geography of U.S. renewable-energy projects

## Contents

```
code/estimate_tables.py   Re-estimation script
data/                     Derived public-data input files
output/tables/            Generated output tables (created on run)
requirements.txt          Pinned package versions
```

## Run

```bash
pip install -r requirements.txt
python code/estimate_tables.py
```

## Outputs

| File | Content |
| --- | --- |
| table_2_common_population_stage_associations.csv | Additive stage decomposition (any entry, entry without operation, entry with operation, difference) |
| table_3_conditional_realization_intensity.csv | Conditional realization-intensity models with minimum detectable effects |
| table_4_endpoint_status_by_rural_class.csv | Endpoint realization, withdrawal, and unresolved shares by rural class |
| functional_form_checks.csv | Logit average marginal effects and Poisson incidence-rate ratios |
| cohort_realization_rates.csv | Realized and unresolved shares by cohort |
| early_cohort_conditional_models.csv | Conditional realization models on 2015-2016 cohorts |
| capability_correlation.csv | County-level attraction vs realization capability correlation |
| linkage_screen_summary.csv | EIA linkage-screen class counts and capacity ratios |

## Inputs

| File | Unit | Rows |
| --- | --- | --- |
| county_technology_cohort_stage_panel_revised.csv | County-technology-cohort cell | 31,080 |
| project_direct_conversion_sample_cross_source.csv | Queue project | 6,988 |
| eia_new_reclassification_linkage_screen_50.csv | EIA-only reclassified project | 50 |

All inputs are derived from public sources (LBNL interconnection queues, EIA-860, NREL supply curves, HIFLD transmission layers, USDA ERS remoteness classifications).

## Update: distinct-capability and geography-coincidence analyses
`code/estimate_tables.py` now also writes, to `output/tables/capability_correlation.csv`:
- county-level attraction-vs-realization capability correlation (resolved-only thresholds 1/3/5);
- an all-projects robustness treating unresolved projects as not-yet-realized (thresholds 3/5);
- the raw county breadth vs realized-share correlation;
- the county proposed-vs-realized capacity (MW) Spearman rank correlation (geography-coincidence test);
- a 2,000-draw county bootstrap 95% interval for the at-least-3-resolved capability correlation.
Main-text Figure 1 (attraction vs conditional-realization coefficients) and Figure 2 (county proposed vs realized capacity) are in `output/figures/`.
