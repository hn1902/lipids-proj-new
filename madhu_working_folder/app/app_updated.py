#!/usr/bin/env python

import seaborn as sns
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import functions
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.express as px
from matplotlib.colors import LogNorm

from shiny import App, render, ui, reactive

# ============================================
# UI Definition
# ============================================
app_ui = ui.page_fluid(
    ui.navset_tab(
        # Upload Data Tab
        ui.nav_panel("Upload Data",
            ui.layout_columns(
                ui.card(
                    ui.input_file("pos_data", "Select data files", accept=[".csv"], multiple=True)
                ),
                ui.card(
                    ui.input_file("header", "Select header file (optional)", 
                                accept=[".csv"], multiple=False)
                ),
                ui.card(
                    ui.input_select(
                        "agg_method",
                        "Aggregation of replicates",
                        {"mean": "Mean", "median": "Median", "sum": "Sum"},
                        selected="mean"
                    ),
                    ui.input_selectize("filter_lipid_class", 
                                     "Filter by Lipid Class", 
                                     choices=[], 
                                     multiple=True),
                    ui.input_numeric("min_chain_length", "Min Acyl Chain Length", 0),
                    ui.input_numeric("max_unsaturation", "Max Unsaturation", 10),
                    ui.input_checkbox("remove_blank", "Remove blank samples", value=False),
                    ui.input_numeric("blank_threshold", "Blank sample threshold (sum)", 0),
                    ui.input_action_button("reset_filters", "Reset Filters")
                ),
                col_widths=(4, 4, 4)
            ),
            
            ui.layout_columns(
                ui.card(
                    ui.output_text_verbatim("warning_message")
                ),
                col_widths=(12,)
            ),
            
            ui.navset_card_tab(
                ui.nav_panel("Filtered Data",
                    ui.layout_columns(
                        ui.card(
                            ui.output_data_frame("filtered_df")
                        ),
                        ui.card(
                            ui.download_button("download_filtered_df", "Download filtered data (CSV)")
                        ),
                        col_widths=(9, 3)
                    )
                ),
                ui.nav_panel("Uploaded Data",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.input_numeric("num_idx", "Number of index columns", 1),
                            ui.input_select("idx_col", "Index column", choices=[]),
                            ui.input_select("main_col_lvl", "Main column level", choices=[]),
                            ui.input_select("cohort_lvl", "Cohort level", choices=[]),
                            ui.input_select("drop_cols", "Columns to drop", choices=[], multiple=True)
                        ),
                        ui.output_data_frame("raw_df")
                    )
                )
            )
        ),
        
        # PCA Tab
        ui.nav_panel("PCA",
            ui.layout_columns(
                ui.card(
                    ui.card_header('Explained Variance'),
                    ui.output_text_verbatim("ev_text"),
                    ui.output_plot("ev_graph")
                ),
                col_widths=(4, 8)
            ),
            ui.navset_card_tab(
                ui.nav_panel('PCA - 2D',
                    ui.output_plot("pca2d"),
                    ui.download_button("download_pca_scores", "Download PCA scores (CSV)"),
                    ui.download_button("download_pca_ev", "Download explained variance (CSV)")
                ),
                ui.nav_panel('PCA - 3D',
                    ui.output_plot("pca3d")
                )
            )
        ),
        
        # Chain Length Tab
        ui.nav_panel("Chain Length",
            ui.layout_columns(
                ui.card(
                    ui.card_header("Chain Length KDE / Histogram"),
                    ui.output_plot("chain_kde_hist")
                ),
                ui.card(
                    ui.card_header("Chain Length Heatmap (Z-score)"),
                    ui.output_plot("chain_heatmap_z")
                ),
                col_widths=(6, 6)
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header('Chain Length Correlation Matrix'),
                    ui.output_plot("chain_corr")
                ),
                col_widths=(12,)
            )
        ),
        
        # Unsaturation Tab
        ui.nav_panel("Unsaturation",
            ui.layout_columns(
                ui.card(
                    ui.card_header("KDE & Histogram (Unsaturation weighted by abundance)"),
                    ui.output_plot("kde_hist_unsat")
                ),
                ui.card(
                    ui.card_header("Heatmap (Z-score of Unsaturation Levels)"),
                    ui.output_plot("heatmap_unsat_z")
                ),
                col_widths=(6, 6)
            )
        ),
        
        # Head Group Tab
        ui.nav_panel("Head Group",
            ui.layout_columns(
                ui.card(
                    ui.card_header("Donut Chart (Normalized)"),
                    ui.output_plot("donut_chart")
                ),
                ui.card(
                    ui.card_header("Heatmap (Z-score of Head Groups)"),
                    ui.output_plot("heatmap_z")
                ),
                col_widths=(6, 6)
            )
        ),
        
        # Statistics Tab
        ui.nav_panel("Statistics",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("stat_var", "Variable to test", choices=[], selected="Head Group 2"),
                    ui.input_numeric("alpha", "Alpha (significance level)", value=0.05),
                    ui.input_selectize("posthoc_level", "Select level for post-hoc", choices=[])
                ),
                ui.card(
                    ui.card_header("ANOVA Results"),
                    ui.output_data_frame("anova_table")
                ),
                ui.card(
                    ui.card_header("Pairwise Comparisons (Tukey HSD)"),
                    ui.output_data_frame("posthoc_table"),
                    ui.download_button("download_posthoc", "Download post-hoc results")
                )
            )
        )
    )
)

# ============================================
# Server Logic
# ============================================
def server(input, output, session):
    # Data loading and processing functions
    @reactive.calc
    def df_func():
        """Load and process the main data."""
        if input.pos_data() is None:
            return None
            
        try:
            # Load data
            df_list = []
            for file in input.pos_data():
                df = pd.read_csv(file["datapath"])
                df_list.append(df)
            df = pd.concat(df_list, ignore_index=True)
            
            # Process data based on user inputs
            # (Add your data processing logic here)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    # Warning message
    @output
    @render.text
    def warning_message():
        df = df_func()
        if df is None:
            return "No data loaded. Please upload data files."
        if df.empty:
            return "Warning: The loaded data is empty."
        return ""
    
    # Display filtered data
    @output
    @render.data_frame
    def filtered_df():
        df = df_func()
        if df is not None:
            return df
        return None
    
    # Download filtered data
    @output
    @render.download(filename="filtered_data.csv")
    async def download_filtered_df():
        df = df_func()
        if df is not None:
            return df.to_csv(index=False)
        return ""
    
    # Display raw data
    @output
    @render.data_frame
    def raw_df():
        df = df_func()
        return df
    
    # ============================================
    # PCA Functions
    # ============================================
    @reactive.calc
    def pca_func():
        """Perform PCA on the processed data."""
        df = df_func()
        if df is None:
            return None
        
        # Select only numeric columns for PCA
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None
            
        X = df[numeric_cols]
        
        # Handle missing values
        if X.isnull().any().any():
            X = X.fillna(0)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=min(10, X_scaled.shape[1]))
        pca_result = pca.fit_transform(X_scaled)
        
        # Create a DataFrame with the PCA results
        pca_df = pd.DataFrame(
            data=pca_result[:, :3],  # First 3 principal components
            columns=['PC1', 'PC2', 'PC3'],
            index=df.index
        )
        
        # Add sample names and any grouping variables
        pca_df['Sample'] = df.index
        
        return {
            'pca': pca,
            'pca_df': pca_df,
            'explained_variance': pca.explained_variance_ratio_,
            'features': numeric_cols,
            'loadings': pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=numeric_cols
            )
        }

    @output
    @render.text
    def ev_text():
        pca_result = pca_func()
        if pca_result is None:
            return "No PCA results available."
        ev = pca_result['explained_variance']
        return f"PC1: {ev[0]:.1%}\nPC2: {ev[1]:.1%}\nPC3: {ev[2]:.1%}"

    @output
    @render.plot
    def ev_graph():
        pca_result = pca_func()
        if pca_result is None:
            return None
        
        ev = pca_result['explained_variance']
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(ev) + 1), ev, alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Component')
        return plt.gcf()

    @output
    @render.plot
    def pca2d():
        pca_result = pca_func()
        if pca_result is None:
            return None
        
        pca_df = pca_result['pca_df']
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Sample')
        plt.title('PCA - First Two Principal Components')
        return plt.gcf()

    @output
    @render.download(filename="pca_scores.csv")
    async def download_pca_scores():
        pca_result = pca_func()
        if pca_result is not None:
            return pca_result['pca_df'].to_csv(index=False)
        return ""

    @output
    @render.download(filename="pca_variance.csv")
    async def download_pca_ev():
        pca_result = pca_func()
        if pca_result is not None:
            ev_df = pd.DataFrame({
                'PC': [f'PC{i+1}' for i in range(len(pca_result['explained_variance']))],
                'ExplainedVariance': pca_result['explained_variance'],
                'CumulativeVariance': np.cumsum(pca_result['explained_variance'])
            })
            return ev_df.to_csv(index=False)
        return ""

    @output
    @render.plot
    def pca3d():
        pca_result = pca_func()
        if pca_result is None:
            return None
        
        pca_df = pca_result['pca_df']
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            pca_df['PC1'], 
            pca_df['PC2'], 
            pca_df['PC3'],
            c=range(len(pca_df)),
            cmap='viridis'
        )
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('3D PCA Plot')
        plt.colorbar(scatter, label='Sample Index')
        return fig
    
    # ============================================
    # Chain Length Analysis
    # ============================================
    @output
    @render.plot
    def chain_kde_hist():
        df = df_func()
        if df is None:
            return None
            
        # Add your chain length analysis code here
        plt.figure(figsize=(10, 6))
        # Example: sns.histplot(...)
        plt.title('Chain Length Distribution')
        return plt.gcf()
    
    @output
    @render.plot
    def chain_heatmap_z():
        df = df_func()
        if df is None:
            return None
            
        # Add your chain length heatmap code here
        plt.figure(figsize=(10, 6))
        # Example: sns.heatmap(...)
        plt.title('Chain Length Z-scores')
        return plt.gcf()
    
    @output
    @render.plot
    def chain_corr():
        df = df_func()
        if df is None:
            return None
            
        # Add your correlation analysis code here
        plt.figure(figsize=(10, 8))
        # Example: sns.heatmap(...)
        plt.title('Chain Length Correlations')
        return plt.gcf()
    
    # ============================================
    # Unsaturation Analysis
    # ============================================
    @output
    @render.plot
    def kde_hist_unsat():
        df = df_func()
        if df is None:
            return None
            
        # Add your unsaturation analysis code here
        plt.figure(figsize=(10, 6))
        # Example: sns.histplot(...)
        plt.title('Unsaturation Distribution')
        return plt.gcf()
    
    @output
    @render.plot
    def heatmap_unsat_z():
        df = df_func()
        if df is None:
            return None
            
        # Add your unsaturation heatmap code here
        plt.figure(figsize=(10, 8))
        # Example: sns.heatmap(...)
        plt.title('Unsaturation Z-scores')
        return plt.gcf()
    
    # ============================================
    # Head Group Analysis
    # ============================================
    @output
    @render.plot
    def donut_chart():
        df = df_func()
        if df is None:
            return None
            
        # Add your donut chart code here
        plt.figure(figsize=(8, 8))
        # Example: plt.pie(...)
        plt.title('Head Group Distribution')
        return plt.gcf()
    
    @output
    @render.plot
    def heatmap_z():
        df = df_func()
        if df is None:
            return None
            
        # Add your head group heatmap code here
        plt.figure(figsize=(10, 8))
        # Example: sns.heatmap(...)
        plt.title('Head Group Z-scores')
        return plt.gcf()
    
    # ============================================
    # Statistical Analysis
    # ============================================
    @reactive.calc
    def stats_result():
        """Perform statistical analysis on the selected variable."""
        df = df_func()
        if df is None or input.stat_var() not in df.columns:
            return None, None
            
        # Perform ANOVA
        anova_results = []
        posthoc_results = {}
        
        # For each numeric column, perform ANOVA and post-hoc tests
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # One-way ANOVA
            groups = [group[col].values for name, group in df.groupby(input.stat_var())]
            if len(groups) < 2:
                continue
                
            f_val, p_val = f_oneway(*groups)
            anova_results.append({
                'Variable': col,
                'F-value': f_val,
                'p-value': p_val,
                'Significant': p_val < input.alpha()
            })
            
            # Post-hoc test if significant
            if p_val < input.alpha():
                posthoc = pairwise_tukeyhsd(
                    endog=df[col].dropna(),
                    groups=df[input.stat_var()],
                    alpha=float(input.alpha())
                )
                posthoc_results[col] = pd.DataFrame(
                    data=posthoc._results_table.data[1:],
                    columns=posthoc._results_table.data[0]
                )
        
        anova_df = pd.DataFrame(anova_results)
        return anova_df, posthoc_results
    
    @output
    @render.data_frame
    def anova_table():
        anova_df, _ = stats_result()
        if anova_df is None:
            return None
        return anova_df
    
    @output
    @render.data_frame
    def posthoc_table():
        _, posthoc_results = stats_result()
        if not posthoc_results:
            return None
            
        selected_level = input.posthoc_level()
        if not selected_level or selected_level not in posthoc_results:
            return None
            
        return posthoc_results[selected_level]
    
    @output
    @render.download(filename="posthoc_results.csv")
    async def download_posthoc():
        _, posthoc_results = stats_result()
        if not posthoc_results:
            return ""
            
        selected_level = input.posthoc_level()
        if not selected_level or selected_level not in posthoc_results:
            return ""
            
        return posthoc_results[selected_level].to_csv(index=False)
    
    # Update UI elements
    @reactive.Effect
    def _():
        df = df_func()
        if df is None:
            return
            
        # Update column selectors
        if hasattr(input, 'idx_col'):
            ui.update_select("idx_col", choices=list(df.columns))
            
        if hasattr(input, 'stat_var'):
            # Update statistical analysis variable selector
            ui.update_select("stat_var", choices=list(df.select_dtypes(include=['category', 'object']).columns))
            
        # Update other UI elements as needed
        # ...

# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=8001)
