#!/usr/bin/env python3
"""
Small analysis helpers for chain-length summaries and plots used by the Shiny app.
"""
from typing import Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import functions as fx


def chain_length_group_tables(df_meta: pd.DataFrame, df_sample: pd.DataFrame, df_cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute chain-length grouped tables.

    Returns:
      df_cl: raw summed abundances grouped by Acyl Chain Length (index = chain length, columns = samples/cohorts depending on input)
      df_cl_norm: column-normalized proportions
      df_cl_z: z-score across samples (rows = chain length)
      df_long: long-form merged table suitable for plotting KDE/hist (columns: Sample Name, Acyl Chain Length, Cohort, Abundance)
    """
    # df_sample: sample-level DataFrame with Sample Name column and experiment columns
    # df_cohort: cohort-aggregated DataFrame with columns as cohorts, index Sample Name
    # merge sample-level to get chain lengths
    df_merged = df_meta.merge(df_sample, on="Sample Name")
    df_cl = df_merged.groupby("Acyl Chain Length").sum()

    # df_cl will have abundance columns per experiment; produce cohort-averaged table similar to headgroup logic
    # produce a cohort-aggregated version using df_cohort
    df_cohort_t = df_cohort.T.reset_index().rename(columns={"index": "Sample Name"})
    df_merged_cohort = df_meta.merge(df_cohort_t, on="Sample Name")
    df_cl_cohort = df_merged_cohort.groupby("Acyl Chain Length").sum()

    # normalize columns (per cohort)
    df_cl_norm = df_cl_cohort.div(df_cl_cohort.sum(axis=0), axis=1)

    # z-score per chain length row
    df_cl_z = df_cl_cohort.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    # long form (cohort-level) for KDE/hist plotting
    df_long = df_merged_cohort.melt(id_vars=["Sample Name", "Acyl Chain Length", "Head Group", "Head Group 2", "Unsaturation", "Unsaturation 2"], var_name="Cohort", value_name="Abundance")

    return df_cl_cohort, df_cl_norm, df_cl_z, df_long


def odd_chain_fraction(df_meta: pd.DataFrame, df_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fraction of odd-chain lipids per cohort.

    Returns DataFrame indexed by Cohort with columns ["Odd", "Even", "FractionOdd"]
    """
    df_cohort_t = df_cohort.T.reset_index().rename(columns={"index": "Sample Name"})
    df_merged = df_meta.merge(df_cohort_t, on="Sample Name")
    # melt to long
    df_long = df_merged.melt(id_vars=["Sample Name", "Acyl Chain Length"], var_name="Cohort", value_name="Abundance")
    df_long['Odd'] = df_long['Acyl Chain Length'] % 2 == 1
    # sum abundances by Cohort and oddness
    s = df_long.groupby(['Cohort', 'Odd'])['Abundance'].sum().unstack(fill_value=0)
    
    # Rename columns from boolean to string
    s.rename(columns={True: 'Odd', False: 'Even'}, inplace=True)

    # Ensure both 'Odd' and 'Even' columns exist
    if 'Odd' not in s.columns:
        s['Odd'] = 0
    if 'Even' not in s.columns:
        s['Even'] = 0

    s['FractionOdd'] = s['Odd'] / (s['Odd'] + s['Even']).replace(0, np.nan)
    s.fillna(0, inplace=True)
    return s


def subset_headgroup_by_chain(df_meta: pd.DataFrame, df_sample: pd.DataFrame, df_cohort: pd.DataFrame, condition) -> pd.DataFrame:
    """
    Returns head group distribution (cohort-aggregated) for lipids satisfying the condition on chain length.
    `condition` is a boolean Series aligned to df_meta rows (or a function applied to Acyl Chain Length).
    """
    # df_sample: sample-level with Sample Name
    df = df_sample.copy()
    # ensure Sample Name in df
    df_meta_local = df_meta.copy()
    # build boolean mask by applying condition to df_meta
    mask = condition(df_meta_local['Acyl Chain Length'])
    keep_names = df_meta_local.loc[mask, 'Sample Name']
    # cohort table filtered
    df_cohort_t = df_cohort.T.reset_index().rename(columns={"index": "Sample Name"})
    df_cohort_filtered = df_cohort_t[df_cohort_t['Sample Name'].isin(keep_names)]
    if df_cohort_filtered.empty:
        return pd.DataFrame()
    df_m = df_meta_local.merge(df_cohort_filtered, on='Sample Name')
    df_hg = df_m.groupby('Head Group 2').sum()
    df_hg_norm = df_hg.div(df_hg.sum(axis=0), axis=1)
    return df_hg_norm


def kde_hist_plot(df_long: pd.DataFrame, figsize=(8, 4)):
    """
    Plot KDE and histogram of chain lengths per cohort using abundance as weights.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    cohorts = df_long['Cohort'].unique()
    pal = sns.color_palette('tab10', n_colors=len(cohorts))
    for i, cohort in enumerate(cohorts):
        sub = df_long[df_long['Cohort'] == cohort]
        if sub['Abundance'].sum() == 0:
            continue
        # use repeated samples according to abundance as approximation for KDE if weights unsupported
        try:
            sns.kdeplot(x=sub['Acyl Chain Length'], weights=sub['Abundance'], label=cohort, ax=ax, color=pal[i])
            sns.histplot(x=sub['Acyl Chain Length'], weights=sub['Abundance'], stat='density', alpha=0.2, color=pal[i], ax=ax, bins=range(int(sub['Acyl Chain Length'].min()), int(sub['Acyl Chain Length'].max())+2))
        except Exception:
            # fallback: scatter of means
            ax.scatter(sub['Acyl Chain Length'], sub['Abundance'], label=cohort, color=pal[i], alpha=0.6)
    ax.set_xlabel('Acyl Chain Length')
    ax.set_ylabel('Density (weighted by abundance)')
    ax.legend()
    plt.tight_layout()
    return fig


def holm_sidak_correction(pvals, alpha=0.05):
    """
    Perform Holm–Sidak correction on a list/array of p-values.

    Returns a dict with keys:
      'reject' : boolean array whether null is rejected
      'p_adjusted' : adjusted p-values (for reporting, using a Holm-style step-down approximation)
      'thresholds' : list of thresholds used for comparisons (sorted order)
    """
    pvals = np.asarray(pvals)
    m = len(pvals)
    if m == 0:
        return {'reject': np.array([]), 'p_adjusted': np.array([]), 'thresholds': []}
    # sort p-values
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    reject_sorted = np.zeros(m, dtype=bool)
    thresholds = np.zeros(m)
    # Holm-Sidak step-down thresholds
    for i in range(m):
        thresh = 1 - (1 - alpha) ** (1 / (m - i))
        thresholds[i] = thresh
        if p_sorted[i] <= thresh:
            reject_sorted[i] = True
        else:
            # once one fails in step-down, remaining are not rejected
            break
    # unsort to original order
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    # approximate adjusted p-values by Holm-style multiplication (conservative)
    p_adj_sorted = np.empty(m)
    for i in range(m):
        p_adj_sorted[i] = min(1.0, p_sorted[i] * (m - i))
    p_adjusted = np.empty(m)
    p_adjusted[order] = p_adj_sorted
    return {'reject': reject, 'p_adjusted': p_adjusted, 'thresholds': thresholds}


def anova_per_level(df_meta: pd.DataFrame, df_cohort: pd.DataFrame, var: str, alpha=0.05):
    """
    Run one-way ANOVA across cohorts for each level of `var` (e.g., each Head Group).

    Returns:
      anova_df: DataFrame indexed by level of `var` with columns ['F','PR(>F)','n_groups']
      posthoc_results: dict mapping level -> DataFrame with pairwise comparisons (Group1, Group2, t-stat, pval, p_adj, reject)
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from scipy import stats

    # melt cohort table to long form and merge with metadata
    df_cohort_t = df_cohort.T.reset_index().rename(columns={'index': 'Sample Name'})
    df_long = df_meta.merge(df_cohort_t, on='Sample Name')
    df_long = df_long.melt(id_vars=['Sample Name', var], var_name='Cohort', value_name='Abundance')

    levels = df_long[var].dropna().unique()
    anova_rows = []
    posthoc = {}

    for lvl in levels:
        sub = df_long[df_long[var] == lvl]
        # need at least 2 groups with data
        groups = sub.groupby('Cohort')['Abundance'].apply(list)
        if groups.shape[0] < 2:
            anova_rows.append({'level': lvl, 'F': np.nan, 'PR(>F)': np.nan, 'n_groups': groups.shape[0]})
            posthoc[lvl] = pd.DataFrame()
            continue
        # build formula and run ANOVA
        try:
            model = smf.ols('Abundance ~ C(Cohort)', data=sub).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            F = aov_table.loc['C(Cohort)', 'F'] if 'C(Cohort)' in aov_table.index else np.nan
            pval = aov_table.loc['C(Cohort)', 'PR(>F)'] if 'C(Cohort)' in aov_table.index else np.nan
        except Exception:
            # fallback to scipy one-way ANOVA across groups
            try:
                grp_lists = [g for _, g in sub.groupby('Cohort')['Abundance']]
                F, pval = stats.f_oneway(*grp_lists)
            except Exception:
                F, pval = np.nan, np.nan

        anova_rows.append({'level': lvl, 'F': F, 'PR(>F)': pval, 'n_groups': groups.shape[0]})

        # if significant, run pairwise t-tests and Holm–Sidak correction
        if not np.isnan(pval) and pval <= alpha:
            cohort_names = groups.index.tolist()
            pvals = []
            pairs = []
            tstats = []
            for i in range(len(cohort_names)):
                for j in range(i+1, len(cohort_names)):
                    g1 = groups[cohort_names[i]]
                    g2 = groups[cohort_names[j]]
                    try:
                        tstat, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
                    except Exception:
                        tstat, p = np.nan, np.nan
                    pvals.append(p)
                    pairs.append((cohort_names[i], cohort_names[j]))
                    tstats.append(tstat)
            if len(pvals) > 0:
                res = holm_sidak_correction(pvals, alpha=alpha)
                p_adjusted = res['p_adjusted']
                reject = res['reject']
                comp_rows = []
                for k, (g1, g2) in enumerate(pairs):
                    comp_rows.append({'Group1': g1, 'Group2': g2, 't-stat': tstats[k], 'pval': pvals[k], 'p_adj': p_adjusted[k], 'reject': bool(reject[k])})
                posthoc[lvl] = pd.DataFrame(comp_rows)
            else:
                posthoc[lvl] = pd.DataFrame()
        else:
            posthoc[lvl] = pd.DataFrame()

    anova_df = pd.DataFrame(anova_rows).set_index('level')
    return anova_df, posthoc


def run_dunnett_test(df_meta: pd.DataFrame, df_cohort: pd.DataFrame, var: str, control_cohort: str):
    """
    Run Dunnett's test to compare all cohorts against a control cohort for each level of 'var'.
    
    Returns a dictionary mapping level -> Dunnett test result object or DataFrame.
    """
    # Prepare long form data
    df_cohort_t = df_cohort.T.reset_index().rename(columns={'index': 'Sample Name'})
    df_long = df_meta.merge(df_cohort_t, on='Sample Name')
    df_long = df_long.melt(id_vars=['Sample Name', var], var_name='Cohort', value_name='Abundance')
    
    levels = df_long[var].dropna().unique()
    results = {}
    
    for lvl in levels:
        sub = df_long[df_long[var] == lvl]
        groups = sub.groupby('Cohort')['Abundance'].apply(list)
        
        if control_cohort not in groups.index or len(groups) < 2:
            results[lvl] = None
            continue
            
        control_data = groups[control_cohort]
        other_data = [groups[c] for c in groups.index if c != control_cohort]
        
        try:
            # Dunnett's test: compare each other group to the control
            res = stats.dunnett(*other_data, control=control_data)
            
            # format into a readable dataframe
            comp_names = [c for c in groups.index if c != control_cohort]
            df_res = pd.DataFrame({
                'Comparison': [f"{c} vs {control_cohort}" for c in comp_names],
                'Statistic': res.statistic,
                'p-value': res.pvalue
            })
            results[lvl] = df_res
        except Exception:
            results[lvl] = None
            
    return results


def run_two_way_anova(df_meta: pd.DataFrame, df_cohort: pd.DataFrame, var1: str, var2: str):
    """
    Run a Two-Way ANOVA analyzing the effect of var1 and var2 on lipid abundance.
    Typically var1 = 'Cohort' and var2 = 'Head Group' or 'Acyl Chain Length'.
    """
    df_cohort_t = df_cohort.T.reset_index().rename(columns={'index': 'Sample Name'})
    df_long = df_meta.merge(df_cohort_t, on='Sample Name')
    # var2 is usually a lipid metadata field, var1 is 'Cohort'
    df_melt = df_long.melt(id_vars=['Sample Name', var2], var_name='Cohort', value_name='Abundance')
    
    # Formula: Abundance ~ C(Cohort) + C(var2) + C(Cohort):C(var2)
    formula = f'Abundance ~ C(Cohort) + C({var2}) + C(Cohort):C({var2})'
    
    try:
        model = smf.ols(formula, data=df_melt).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        return aov_table
    except Exception as e:
        print(f"Two-Way ANOVA failed: {e}")
        return None


def unsaturation_group_tables(df_meta: pd.DataFrame, df_sample: pd.DataFrame, df_cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute unsaturation-grouped tables (mirror of chain_length_group_tables).

    Returns:
      df_unsat: raw summed abundances grouped by Unsaturation (index = unsaturation level, columns = cohorts)
      df_unsat_norm: column-normalized proportions
      df_unsat_z: z-score across cohorts
      df_long: long-form merged table for plotting (columns: Sample Name, Unsaturation, Cohort, Abundance)
    """
    # Prepare cohort-aggregated version
    df_cohort_t = df_cohort.T.reset_index().rename(columns={"index": "Sample Name"})
    df_merged_cohort = df_meta.merge(df_cohort_t, on="Sample Name")
    
    # Group by unsaturation
    df_unsat = df_merged_cohort.groupby("Unsaturation").sum()
    
    # Normalize columns (per cohort)
    df_unsat_norm = df_unsat.div(df_unsat.sum(axis=0), axis=1)
    
    # Z-score per unsaturation row
    df_unsat_z = df_unsat.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    # Long form for KDE/hist plotting
    df_long = df_merged_cohort.melt(
        id_vars=["Sample Name", "Unsaturation", "Head Group", "Head Group 2", "Acyl Chain Length"],
        var_name="Cohort",
        value_name="Abundance"
    )
    
    return df_unsat, df_unsat_norm, df_unsat_z, df_long


def kde_hist_plot_unsat(df_long: pd.DataFrame, figsize=(8, 4)):
    """
    Plot KDE and histogram of unsaturation per cohort using abundance as weights.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    cohorts = df_long['Cohort'].unique()
    pal = sns.color_palette('tab10', n_colors=len(cohorts))
    for i, cohort in enumerate(cohorts):
        sub = df_long[df_long['Cohort'] == cohort]
        if sub['Abundance'].sum() == 0:
            continue
        try:
            sns.kdeplot(x=sub['Unsaturation'], weights=sub['Abundance'], label=cohort, ax=ax, color=pal[i])
            sns.histplot(
                x=sub['Unsaturation'], 
                weights=sub['Abundance'], 
                stat='density', 
                alpha=0.2, 
                color=pal[i], 
                ax=ax, 
                bins=range(int(sub['Unsaturation'].min()), int(sub['Unsaturation'].max())+2)
            )
        except Exception:
            # fallback: scatter of means
            ax.scatter(sub['Unsaturation'], sub['Abundance'], label=cohort, color=pal[i], alpha=0.6)
    ax.set_xlabel('Unsaturation (# double bonds)')
    ax.set_ylabel('Density (weighted by abundance)')
    ax.legend()
    plt.tight_layout()
    return fig


def subset_headgroup_by_unsat(df_meta: pd.DataFrame, df_cohort: pd.DataFrame, condition) -> pd.DataFrame:
    """
    Returns head group distribution (cohort-aggregated) for lipids satisfying the condition on unsaturation.
    `condition` is a function applied to Unsaturation values.
    """
    df_meta_local = df_meta.copy()
    mask = condition(df_meta_local['Unsaturation'])
    keep_names = df_meta_local.loc[mask, 'Sample Name']
    
    df_cohort_t = df_cohort.T.reset_index().rename(columns={"index": "Sample Name"})
    df_cohort_filtered = df_cohort_t[df_cohort_t['Sample Name'].isin(keep_names)]
    
    if df_cohort_filtered.empty:
        return pd.DataFrame()
    
    df_m = df_meta_local.merge(df_cohort_filtered, on='Sample Name')
    df_hg = df_m.groupby('Head Group 2').sum()
    df_hg_norm = df_hg.div(df_hg.sum(axis=0), axis=1)
    return df_hg_norm


def lipid_class_tables(df_meta: pd.DataFrame, df_cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute lipid class-level aggregations.

    Returns:
      df_lc: raw summed abundances grouped by Lipid Class (index = class, columns = cohorts)
      df_lc_norm: column-normalized proportions (for pie charts and heatmaps)
      df_lc_z: z-score across cohorts
    """
    df_cohort_t = df_cohort.T.reset_index().rename(columns={"index": "Sample Name"})
    df_merged = df_meta.merge(df_cohort_t, on="Sample Name")
    
    
    # Group by Lipid Class
    if 'Lipid Class' not in df_merged.columns:
        # Fallback: if Lipid Class not available, use Head Group as proxy
        df_lc = df_merged.groupby('Head Group').sum()
    else:
        df_lc = df_merged.groupby('Lipid Class').sum()
    
    # Normalize columns (per cohort, produces proportions)
    df_lc_norm = df_lc.div(df_lc.sum(axis=0), axis=1)
    
    # Z-score per class row
    df_lc_z = df_lc.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    return df_lc, df_lc_norm, df_lc_z


def run_per_point_ttest(df_meta: pd.DataFrame, df_cohort: pd.DataFrame, var: str, control_cohort: str):
    """
    For each level of 'var', perform a t-test between the control cohort and all other cohorts.
    Returns:
      results_df: DataFrame with Cohort, Level, MeanDifference, PValue, and Significance.
    """
    from scipy import stats
    
    # Prepare long form data
    df_cohort_t = df_cohort.T.reset_index().rename(columns={'index': 'Sample Name'})
    df_long = df_meta.merge(df_cohort_t, on='Sample Name')
    df_long = df_long.melt(id_vars=['Sample Name', var], var_name='Cohort', value_name='Abundance')
    
    levels = df_long[var].dropna().unique()
    cohorts = df_long['Cohort'].unique()
    
    rows = []
    for lvl in levels:
        sub = df_long[df_long[var] == lvl]
        groups = sub.groupby('Cohort')['Abundance'].apply(list)
        
        if control_cohort not in groups.index:
            continue
            
        ctrl_vals = groups[control_cohort]
        for c in cohorts:
            if c == control_cohort or c not in groups.index:
                continue
            
            test_vals = groups[c]
            try:
                tstat, pval = stats.ttest_ind(ctrl_vals, test_vals, equal_var=False, nan_policy='omit')
                mean_diff = np.mean(test_vals) - np.mean(ctrl_vals)
                rows.append({
                    'Cohort': c,
                    'Level': lvl,
                    'MeanDiff': mean_diff,
                    'PValue': pval,
                    'Significant': pval < 0.05
                })
            except Exception:
                pass
                
    return pd.DataFrame(rows)

def gaus(x, a, x0, sigma):
    """Gaussian function."""
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def fit_single_gaussian(x, y):
    """
    Fits a single Gaussian to the data.
    
    Returns:
        popt: [amplitude, mean, sigma]
        r_squared: coefficient of determination
    """
    if len(x) < 3 or np.sum(y) == 0:
        return None, 0
    
    # Initial guesses
    n = np.sum(y)
    mean_guess = np.sum(x * y) / n
    sigma_guess = np.sqrt(np.abs(np.sum(y * (x - mean_guess) ** 2) / n))
    a_guess = np.max(y)
    
    try:
        popt, pcov = curve_fit(gaus, x, y, p0=[a_guess, mean_guess, sigma_guess], maxfev=2000)
        
        # Calculate R-squared
        residuals = y - gaus(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return popt, r_squared
    except Exception:
        return None, 0
