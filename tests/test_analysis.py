import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app import analysis


def make_sample_dfs():
    # sample-level df: Sample Name index with two experiment columns
    df_sample = pd.DataFrame({
        'Sample Name': ['PC 34 1', 'PC 35 2', 'PE 50 0'],
        'exp1': [10, 0, 5],
        'exp2': [5, 0, 5]
    }).set_index('Sample Name')
    df_sample.reset_index(inplace=True)
    # cohort-aggregated: columns are cohort names, index Sample Name
    df_cohort = pd.DataFrame({
        'Sample Name': ['PC 34 1', 'PC 35 2', 'PE 50 0'],
        'CAS9.1': [10, 0, 5],
        'CAV.1': [5, 0, 5]
    }).set_index('Sample Name')
    # meta
    df_meta = pd.DataFrame([
        {'Sample Name': 'PC 34 1', 'Head Group': 'PC', 'Acyl Chain Length': 34, 'Unsaturation': 1, 'Head Group 2': 'PC', 'Unsaturation 2': 1},
        {'Sample Name': 'PC 35 2', 'Head Group': 'PC', 'Acyl Chain Length': 35, 'Unsaturation': 2, 'Head Group 2': 'PC', 'Unsaturation 2': 2},
        {'Sample Name': 'PE 50 0', 'Head Group': 'PE', 'Acyl Chain Length': 50, 'Unsaturation': 0, 'Head Group 2': 'PE', 'Unsaturation 2': 0},
    ])
    # df_sample expected in analysis functions: sample-level wide with Sample Name and cohort columns
    df_sample_wide = pd.DataFrame({
        'Sample Name': ['PC 34 1', 'PC 35 2', 'PE 50 0'],
        'exp1': [10, 0, 5],
        'exp2': [5, 0, 5]
    })
    return df_meta, df_sample_wide, df_cohort


def test_chain_length_group_tables_basic():
    df_meta, df_sample, df_cohort = make_sample_dfs()
    df_cl, df_cl_norm, df_cl_z, df_long = analysis.chain_length_group_tables(df_meta, df_sample, df_cohort)
    # df_cl should have index with chain lengths 34,35,50
    assert set(df_cl.index) == {34,35,50}
    # normalized table should sum columns to ~1
    col_sums = df_cl_norm.sum(axis=0)
    assert np.allclose(col_sums.values, np.ones_like(col_sums.values))
    # long-form should have expected columns
    assert 'Cohort' in df_long.columns and 'Acyl Chain Length' in df_long.columns


def test_odd_chain_fraction():
    df_meta, df_sample, df_cohort = make_sample_dfs()
    s = analysis.odd_chain_fraction(df_meta, df_cohort)
    # Should contain cohorts as index
    assert 'CAS9.1' in s.index or 'CAV.1' in s.index
    # Fractions between 0 and 1
    assert (s['FractionOdd'] >= 0).all() and (s['FractionOdd'] <= 1).all()


def test_subset_headgroup_by_chain():
    df_meta, df_sample, df_cohort = make_sample_dfs()
    df_hg_norm = analysis.subset_headgroup_by_chain(df_meta, df_sample, df_cohort, lambda x: x >= 50)
    # For >=50 we expect a non-empty result with PE present
    assert 'PE' in df_hg_norm.index


def test_kde_hist_plot_returns_figure():
    df_meta, df_sample, df_cohort = make_sample_dfs()
    df_cl, df_cl_norm, df_cl_z, df_long = analysis.chain_length_group_tables(df_meta, df_sample, df_cohort)
    fig = analysis.kde_hist_plot(df_long)
    assert isinstance(fig, matplotlib.figure.Figure)



def test_no_exceptions_when_empty_subset():
    # condition that matches nothing
    df_meta, df_sample, df_cohort = make_sample_dfs()
    df_hg_norm = analysis.subset_headgroup_by_chain(df_meta, df_sample, df_cohort, lambda x: x > 1000)
    assert isinstance(df_hg_norm, pd.DataFrame)
