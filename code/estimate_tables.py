#!/usr/bin/env python3
"""Re-estimate manuscript Tables 2, 3, and 4 and the functional-form and
power checks from the bundled derived public-data files.

Inputs (data/):
    county_technology_cohort_stage_panel_revised.csv
    project_direct_conversion_sample_cross_source.csv
    eia_new_reclassification_linkage_screen_50.csv

Outputs (output/tables/):
    table_2_common_population_stage_associations.csv
    table_3_conditional_realization_intensity.csv
    table_4_endpoint_status_by_rural_class.csv
    functional_form_checks.csv
    cohort_realization_rates.csv
    early_cohort_conditional_models.csv
    capability_correlation.csv
    linkage_screen_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Power factor for the minimum detectable effect at 80% power, two-sided 5%.
MDE_FACTOR = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)

# Attraction-model right-hand side.
ATTRACTION_RHS = (
    "resource_quality_z + grid_access_z + remoteness_z + grid_remote "
    "+ resource_grid + resource_remote + potential_capacity_mw_z "
    "+ area_km2 + population_2010 + log_stock + recent2_mw"
)
CONTROLS_Z = ["area_km2", "population_2010", "recent2_mw"]
KEY_VARS = [
    "queue_any", "resource_quality_z", "grid_access_z", "remoteness_z",
    "grid_remote", "resource_grid", "resource_remote",
    "potential_capacity_mw_z", "log_stock", "sty", "fips",
] + CONTROLS_Z
FOCAL = ["resource_quality_z", "grid_access_z", "grid_remote"]


def standardize(frame, cols):
    out = frame.copy()
    for col in cols:
        out[col] = (out[col] - out[col].mean()) / out[col].std()
    return out


def cluster_ols(formula, frame, cluster_col):
    return smf.ols(formula, data=frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame[cluster_col]}
    )


def coef_row(result, variable, scale=1.0):
    coef = result.params[variable] * scale
    low, high = (result.conf_int().loc[variable] * scale).tolist()
    return coef, low, high, result.pvalues[variable]


def load_inputs():
    panel = pd.read_csv(DATA / "county_technology_cohort_stage_panel_revised.csv")
    proj = pd.read_csv(DATA / "project_direct_conversion_sample_cross_source.csv")
    proj["cohort"] = proj["cohort"].astype(int)
    return panel, proj


def build_attraction_sample(panel, proj):
    # Cell-level realized indicator for the additive stage decomposition.
    cell_op = (
        proj.groupby(["fips", "tech", "cohort"])["realized_cross_source"]
        .max()
        .reset_index()
        .rename(columns={"realized_cross_source": "cell_any_op"})
    )
    frame = panel.merge(cell_op, on=["fips", "tech", "cohort"], how="left")
    frame["cell_any_op"] = frame["cell_any_op"].fillna(0)
    frame["entry_op"] = ((frame["queue_any"] == 1) & (frame["cell_any_op"] == 1)).astype(int)
    frame["entry_noop"] = ((frame["queue_any"] == 1) & (frame["cell_any_op"] == 0)).astype(int)
    frame["diff_outcome"] = frame["entry_noop"] - frame["entry_op"]
    frame = frame.dropna(subset=KEY_VARS).copy()
    frame = standardize(frame, CONTROLS_Z)
    return frame


def estimate_table2(frame):
    # Linear probability decomposition on a common county-technology-cohort sample.
    outcomes = {
        "any_entry": "queue_any",
        "entry_no_op": "entry_noop",
        "entry_op": "entry_op",
        "no_op_minus_op": "diff_outcome",
    }
    rows = []
    for label, outcome in outcomes.items():
        frame["_y"] = frame[outcome] * 100.0
        res = cluster_ols("_y ~ " + ATTRACTION_RHS + " + C(sty)", frame, "fips")
        for var in FOCAL:
            coef, low, high, pval = coef_row(res, var)
            rows.append({
                "outcome": label, "variable": var,
                "coef_pp": round(coef, 2),
                "ci_low": round(low, 2), "ci_high": round(high, 2),
                "p_value": pval, "n": int(res.nobs),
            })
    return pd.DataFrame(rows)


def estimate_functional_forms(frame):
    # Logit average marginal effects and Poisson incidence-rate ratios with
    # state, technology, and cohort fixed effects.
    fe = " + C(state) + C(tech) + C(cohort)"
    rows = []

    logit = smf.logit("queue_any ~ " + ATTRACTION_RHS + fe, data=frame).fit(disp=0, maxiter=300)
    ame = logit.get_margeff(at="overall").summary_frame()
    for var in FOCAL:
        rows.append({
            "form": "logit_ame_pp", "variable": var,
            "estimate": round(ame.loc[var, "dy/dx"] * 100, 2),
            "p_value": ame.loc[var, "Pr(>|z|)"],
        })

    pois = smf.poisson("queue_count ~ " + ATTRACTION_RHS + fe, data=frame).fit(disp=0, maxiter=300)
    for var in FOCAL:
        rows.append({
            "form": "poisson_irr", "variable": var,
            "estimate": round(float(np.exp(pois.params[var])), 3),
            "p_value": pois.pvalues[var],
        })
    return pd.DataFrame(rows)


def build_conditional_cells(panel, proj):
    proj = proj.copy()
    proj["resolved"] = proj["resolved_cross_source"]
    proj["realized"] = proj["realized_cross_source"]
    proj["withdrawn"] = proj["withdrawn_cross_source"]
    proj["mw"] = proj["component_mw"]

    grouped = proj.groupby(["fips", "tech", "cohort"])
    agg = grouped.agg(
        n=("realized", "size"),
        n_real=("realized", "sum"),
        n_resolved=("resolved", "sum"),
        mw_total=("mw", "sum"),
    ).reset_index()
    agg["real_mw"] = grouped.apply(
        lambda g: float((g["mw"] * g["realized"]).sum()), include_groups=False
    ).values
    agg["resolved_real"] = grouped.apply(
        lambda g: float((g["resolved"] * g["realized"]).sum()), include_groups=False
    ).values
    agg["resolved_wd"] = grouped.apply(
        lambda g: float((g["resolved"] * g["withdrawn"]).sum()), include_groups=False
    ).values

    agg["any_op"] = (agg["n_real"] > 0).astype(float) * 100.0
    agg["proj_share"] = agg["n_real"] / agg["n"] * 100.0
    agg["resolved_share"] = np.where(
        agg["n_resolved"] > 0, agg["resolved_real"] / agg["n_resolved"] * 100.0, np.nan
    )
    agg["mw_share"] = np.where(
        agg["mw_total"] > 0, agg["real_mw"] / agg["mw_total"] * 100.0, np.nan
    )
    agg["wd_share"] = np.where(
        agg["n_resolved"] > 0, agg["resolved_wd"] / agg["n_resolved"] * 100.0, np.nan
    )

    cells = panel.merge(agg, on=["fips", "tech", "cohort"], how="left")
    cells = cells[cells["queue_any"] == 1].copy()
    return cells


def estimate_table3(cells):
    # Conditional realization-intensity models on queue-positive cells with the
    # minimum detectable effect at 80% power for the grid and grid-by-remoteness terms.
    rhs = "resource_quality_z + grid_access_z + remoteness_z + grid_remote + resource_grid + C(sty)"
    targets = {
        "any_direct_operation": "any_op",
        "project_direct_operation_share": "proj_share",
        "resolved_project_direct_operation_share": "resolved_share",
        "mw_weighted_direct_operation_share": "mw_share",
        "resolved_withdrawal_share": "wd_share",
    }
    rows = []
    for label, outcome in targets.items():
        sub = cells.dropna(subset=[outcome, "grid_access_z", "grid_remote",
                                   "resource_quality_z", "remoteness_z", "sty"]).copy()
        sub["_y"] = sub[outcome]
        res = cluster_ols("_y ~ " + rhs, sub, "fips")
        grid_c, grid_lo, grid_hi, _ = coef_row(res, "grid_access_z")
        gxr_c, gxr_lo, gxr_hi, _ = coef_row(res, "grid_remote")
        rows.append({
            "outcome": label,
            "mean_pct": round(sub["_y"].mean(), 2),
            "n_cells": int(res.nobs),
            "n_clusters": int(sub["fips"].nunique()),
            "grid_coef": round(grid_c, 2), "grid_ci_low": round(grid_lo, 2), "grid_ci_high": round(grid_hi, 2),
            "grid_remote_coef": round(gxr_c, 2), "grid_remote_ci_low": round(gxr_lo, 2), "grid_remote_ci_high": round(gxr_hi, 2),
            "mde_grid_pp": round(MDE_FACTOR * res.bse["grid_access_z"], 2),
            "mde_grid_remote_pp": round(MDE_FACTOR * res.bse["grid_remote"], 2),
        })
    return pd.DataFrame(rows)


def estimate_table4(proj):
    order = ["Metropolitan", "Nonmetro adjacent", "Nonmetro nonadjacent"]
    rows = []
    for cls in order:
        sub = proj[proj["rural_class"] == cls]
        resolved = sub[sub["resolved_cross_source"] == 1]
        rows.append({
            "class": cls,
            "n": int(len(sub)),
            "realized_pct": round(sub["realized_cross_source"].mean() * 100, 2),
            "withdrawn_pct": round(sub["withdrawn_cross_source"].mean() * 100, 2),
            "unresolved_pct": round(sub["unresolved_cross_source"].mean() * 100, 2),
            "realized_among_resolved_pct": round(resolved["realized_cross_source"].mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def linkage_summary():
    audit = pd.read_csv(DATA / "eia_new_reclassification_linkage_screen_50.csv")
    summary = (
        audit.groupby("linkage_screen_class")
        .agg(projects=("uid", "count"),
             mean_capacity_ratio=("capacity_ratio", "mean"),
             min_capacity_ratio=("capacity_ratio", "min"))
        .reset_index()
    )
    expected = {"high_confidence": 27, "phase_or_partial": 14, "weak": 9}
    counts = audit["linkage_screen_class"].str.lower().value_counts().to_dict()
    for key, val in expected.items():
        assert int(counts.get(key, 0)) == val, f"linkage class mismatch: {key}"
    return summary


def estimate_cohort_stratified(panel, proj):
    # Realization rates by cohort and conditional models on early, largely
    # resolved cohorts, to show the conditional null is not a censoring artifact.
    rate_rows = []
    for cohort in sorted(proj["cohort"].unique()):
        sub = proj[proj["cohort"] == cohort]
        rate_rows.append({
            "cohort": int(cohort), "n": int(len(sub)),
            "realized_pct": round(sub["realized_cross_source"].mean() * 100, 2),
            "unresolved_pct": round(sub["unresolved_cross_source"].mean() * 100, 2),
            "resolved_pct": round(sub["resolved_cross_source"].mean() * 100, 2),
        })

    early = [2015, 2016]
    p = proj[proj["cohort"].isin(early)].copy()
    grouped = p.groupby(["fips", "tech", "cohort"])
    agg = grouped.agg(
        n=("realized_cross_source", "size"),
        n_real=("realized_cross_source", "sum"),
        mw_total=("component_mw", "sum"),
    ).reset_index()
    agg["real_mw"] = grouped.apply(
        lambda g: float((g["component_mw"] * g["realized_cross_source"]).sum()),
        include_groups=False,
    ).values
    agg["any_op"] = (agg["n_real"] > 0).astype(float) * 100.0
    agg["proj_share"] = agg["n_real"] / agg["n"] * 100.0
    agg["mw_share"] = np.where(agg["mw_total"] > 0, agg["real_mw"] / agg["mw_total"] * 100.0, np.nan)
    cells = panel[panel["cohort"].isin(early)].merge(agg, on=["fips", "tech", "cohort"], how="left")
    cells = cells[cells["queue_any"] == 1].copy()

    rhs = "resource_quality_z + grid_access_z + remoteness_z + grid_remote + resource_grid + C(sty)"
    model_rows = []
    for label, outcome in [("project_op_share", "proj_share"),
                           ("mw_op_share", "mw_share"),
                           ("any_operation", "any_op")]:
        sub = cells.dropna(subset=[outcome, "grid_access_z", "grid_remote",
                                   "resource_quality_z", "remoteness_z", "sty"]).copy()
        sub["_y"] = sub[outcome]
        res = cluster_ols("_y ~ " + rhs, sub, "fips")
        grid_c, grid_lo, grid_hi, _ = coef_row(res, "grid_access_z")
        gxr_c, gxr_lo, gxr_hi, _ = coef_row(res, "grid_remote")
        model_rows.append({
            "outcome": label, "n_cells": int(res.nobs),
            "mean_pct": round(sub["_y"].mean(), 2),
            "grid_coef": round(grid_c, 2), "grid_ci_low": round(grid_lo, 2), "grid_ci_high": round(grid_hi, 2),
            "grid_remote_coef": round(gxr_c, 2), "grid_remote_ci_low": round(gxr_lo, 2), "grid_remote_ci_high": round(gxr_hi, 2),
            "mde_grid_pp": round(MDE_FACTOR * res.bse["grid_access_z"], 2),
            "mde_grid_remote_pp": round(MDE_FACTOR * res.bse["grid_remote"], 2),
        })
    return pd.DataFrame(rate_rows), pd.DataFrame(model_rows)


def estimate_capability_correlation(panel, proj):
    # County-level correlation between attraction capability and realization
    # capability, after partialling out shared observables and fixed effects.
    frame = build_attraction_sample(panel, proj)
    frame["_y"] = frame["queue_any"] * 100.0
    attr_model = smf.ols("_y ~ " + ATTRACTION_RHS + " + C(sty)", data=frame).fit()
    frame["resid"] = attr_model.resid
    attraction = frame.groupby("fips")["resid"].mean().rename("attraction_cap").reset_index()

    res = proj[proj["resolved_cross_source"] == 1].copy()
    rkey = ["realized_cross_source", "resource_quality_z", "grid_access_z",
            "remoteness_z", "grid_remote", "resource_grid", "sty", "fips"]
    res = res.dropna(subset=rkey).copy()
    res["_y"] = res["realized_cross_source"] * 100.0
    real_model = smf.ols(
        "_y ~ resource_quality_z + grid_access_z + remoteness_z + grid_remote + resource_grid + C(sty)",
        data=res,
    ).fit()
    res["resid"] = real_model.resid
    n_resolved = res.groupby("fips").size().rename("n_resolved")
    realization = (res.groupby("fips")["resid"].mean().rename("realization_cap")
                   .reset_index().merge(n_resolved, on="fips"))

    merged = attraction.merge(realization, on="fips")
    rows = []
    for threshold in [1, 3, 5]:
        sub = merged[merged["n_resolved"] >= threshold]
        pear = stats.pearsonr(sub["attraction_cap"], sub["realization_cap"])
        spear = stats.spearmanr(sub["attraction_cap"], sub["realization_cap"])
        rows.append({
            "sample": f"at_least_{threshold}_resolved", "counties": int(len(sub)),
            "pearson_r": round(float(pear.statistic), 3), "pearson_p": round(float(pear.pvalue), 3),
            "spearman_rho": round(float(spear.statistic), 3), "spearman_p": round(float(spear.pvalue), 3),
        })

    # All-projects robustness: treat unresolved as not-yet-realized (addresses
    # selection on resolution in the realization-capability index).
    allp = proj.dropna(subset=rkey).copy()
    allp["_y"] = allp["realized_cross_source"] * 100.0
    allp["resid"] = smf.ols(
        "_y ~ resource_quality_z + grid_access_z + remoteness_z + grid_remote + resource_grid + C(sty)",
        data=allp,
    ).fit().resid
    n_all = allp.groupby("fips").size().rename("n_all")
    real_all = (allp.groupby("fips")["resid"].mean().rename("realization_cap_all")
                .reset_index().merge(n_all, on="fips"))
    merged_all = attraction.merge(real_all, on="fips")
    for threshold in [3, 5]:
        sub = merged_all[merged_all["n_all"] >= threshold]
        pear = stats.pearsonr(sub["attraction_cap"], sub["realization_cap_all"])
        spear = stats.spearmanr(sub["attraction_cap"], sub["realization_cap_all"])
        rows.append({
            "sample": f"all_projects_unresolved_zero_at_least_{threshold}", "counties": int(len(sub)),
            "pearson_r": round(float(pear.statistic), 3), "pearson_p": round(float(pear.pvalue), 3),
            "spearman_rho": round(float(spear.statistic), 3), "spearman_p": round(float(spear.pvalue), 3),
        })

    raw = proj.groupby("fips").agg(
        breadth=("uid", "count"),
        real_res=("realized_cross_source", "sum"),
        n_res=("resolved_cross_source", "sum"),
    ).reset_index()
    raw = raw[raw["n_res"] >= 3].copy()
    raw["real_share"] = raw["real_res"] / raw["n_res"]
    pear = stats.pearsonr(np.log(raw["breadth"]), raw["real_share"])
    spear = stats.spearmanr(raw["breadth"], raw["real_share"])
    rows.append({
        "sample": "raw_breadth_vs_realized_share_at_least_3", "counties": int(len(raw)),
        "pearson_r": round(float(pear.statistic), 3), "pearson_p": round(float(pear.pvalue), 3),
        "spearman_rho": round(float(spear.statistic), 3), "spearman_p": round(float(spear.pvalue), 3),
    })

    # County proposed vs realized capacity (MW): direct geography-coincidence test.
    cap = proj.groupby("fips").apply(
        lambda d: pd.Series({
            "prop_mw": d["component_mw"].sum(),
            "real_mw": float((d["component_mw"] * d["realized_cross_source"]).sum()),
        }), include_groups=False).reset_index()
    cap = cap[cap["prop_mw"] > 0]
    spear = stats.spearmanr(cap["prop_mw"], cap["real_mw"])
    rows.append({
        "sample": "county_proposed_vs_realized_MW", "counties": int(len(cap)),
        "pearson_r": float("nan"), "pearson_p": float("nan"),
        "spearman_rho": round(float(spear.statistic), 3), "spearman_p": round(float(spear.pvalue), 6),
    })

    # County bootstrap 95% CI for the at-least-3-resolved capability correlation.
    base = merged[merged["n_resolved"] >= 3]
    pairs = base[["attraction_cap", "realization_cap"]].to_numpy()
    rng = np.random.default_rng(42)
    boots = np.array([
        stats.pearsonr(pairs[idx, 0], pairs[idx, 1]).statistic
        for idx in (rng.integers(0, len(pairs), len(pairs)) for _ in range(2000))
    ])
    rows.append({
        "sample": "bootstrap_ci_at_least_3_resolved", "counties": int(len(base)),
        "pearson_r": round(float(stats.pearsonr(base["attraction_cap"], base["realization_cap"]).statistic), 3),
        "pearson_p": round(float(np.percentile(boots, 2.5)), 3),
        "spearman_rho": round(float(np.percentile(boots, 97.5)), 3),
        "spearman_p": float("nan"),
    })
    return pd.DataFrame(rows)


def main():
    panel, proj = load_inputs()

    attraction = build_attraction_sample(panel, proj)
    table2 = estimate_table2(attraction)
    table2.to_csv(TABLES / "table_2_common_population_stage_associations.csv", index=False)

    forms = estimate_functional_forms(attraction)
    forms.to_csv(TABLES / "functional_form_checks.csv", index=False)

    cells = build_conditional_cells(panel, proj)
    table3 = estimate_table3(cells)
    table3.to_csv(TABLES / "table_3_conditional_realization_intensity.csv", index=False)

    table4 = estimate_table4(proj)
    table4.to_csv(TABLES / "table_4_endpoint_status_by_rural_class.csv", index=False)

    cohort_rates, early_models = estimate_cohort_stratified(panel, proj)
    cohort_rates.to_csv(TABLES / "cohort_realization_rates.csv", index=False)
    early_models.to_csv(TABLES / "early_cohort_conditional_models.csv", index=False)

    capability = estimate_capability_correlation(panel, proj)
    capability.to_csv(TABLES / "capability_correlation.csv", index=False)

    linkage_summary().to_csv(TABLES / "linkage_screen_summary.csv", index=False)

    print(table2.to_string(index=False))
    print(forms.to_string(index=False))
    print(table3.to_string(index=False))
    print(table4.to_string(index=False))
    print(cohort_rates.to_string(index=False))
    print(early_models.to_string(index=False))
    print(capability.to_string(index=False))


if __name__ == "__main__":
    main()
