#!/usr/bin/env python

import seaborn as sns
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import functions # type: ignore

from shiny import reactive
from shiny.express import input, render, ui

ui.page_opts(title="Lipidomics Analysis Pipeline")

ui.nav_spacer()  # Push the navbar items to the right

with ui.nav_panel("Upload Data"):
    with ui.layout_columns(col_widths=(4, 4, 4)):
        with ui.card():
            ui.input_file("pos_data", "Select data files", accept=[".csv"], multiple=True)
        with ui.card():
            ui.input_file(
                "header", "Select header file (optional)", accept=[".csv"], multiple=False
            )
        with ui.card():
            ui.input_select(
                "agg_method",
                "Aggregation of replicates",
                choices={"mean": "Mean", "median": "Median", "sum": "Sum"},
                selected="mean",
            )
            ui.input_selectize("filter_lipid_class", "Filter by Lipid Class", choices=[], multiple=True)
            ui.input_numeric("min_chain_length", "Min Acyl Chain Length", 0)
            ui.input_numeric("max_unsaturation", "Max Unsaturation", 10)
            ui.input_checkbox("remove_blank", "Remove blank samples", value=False)
            ui.input_numeric("blank_threshold", "Blank sample threshold (sum)", 0)
            ui.input_action_button("reset_filters", "Reset Filters")

    with ui.layout_columns(col_widths=(12)):
        with ui.card():
            @render.text
            def warning_message():
                df_result = df_func()
                if df_result is None:
                    return "Warning: No data remains after applying filters."
                df, _ = df_result
                if df.empty:
                    return "Warning: No data remains after applying filters."
                return ""

    with ui.navset_card_underline(title="Dataframes"):   
        @reactive.calc
        def df_parsed_func():
            if input.pos_data() is None or input.main_col_lvl() is None or input.cohort_lvl() is None or input.num_idx() is None or input.num_col_lvls() is None:
                return 
            else:
                df_list = []
                for file in input.pos_data():
                    chunk = pd.read_csv(file["datapath"])
                    df_list.append(chunk)
                df = pd.concat(df_list, ignore_index=True)
                df.set_index(input.idx_col(), inplace=True)
                df.index.name = 'Sample Name'
                df.drop(columns=list(df.columns[:input.num_idx() - 1]), inplace=True) # drop index columns
                
                # Check for drop columns
                drop_cols = input.drop_col()
                if drop_cols:
                    df = df[df.columns[~df.columns.isin(drop_cols)]]
                
                w = int(input.main_col_lvl()) - 2 # check row idx of column
                w2 = int(input.cohort_lvl()) - 2 # check row_idx of cohort column
                df_cohort = df.copy()
                if input.header() is None: # use columns in df
                    if w >= 0: # if w < 0, use original column name
                        df.columns = list(df.iloc[w]) # set column names based on row idx
                    if w2 >= 0:
                        df_cohort.columns = [col.split('.')[0] for col in df_cohort.iloc[w2]] # Extract prefix before decimal
                else: # use submitted header
                    for h in input.header():
                        header = pd.read_csv(h['datapath'])
                        header = header.T
                    df.drop(columns=list(df.columns[~df.columns.isin(header.index)]), inplace=True) #drop any columns not in header
                    df_cohort.drop(columns=list(df_cohort.columns[~df_cohort.columns.isin(header.index)]), inplace=True) #drop any columns not in header
                    if w >= 0:
                        df.rename(columns=header[w], inplace=True) # rename based on column level
                    if w2 >= 0:
                        cohort_names = header[w2]
                        cohort_names = [name.split('.')[0] for name in cohort_names]
                        df_cohort.rename(columns=header[w2], inplace=True) # rename based on column level
                        df_cohort.columns = cohort_names
                
                df = df.iloc[(input.num_col_lvls() - 1) :]  # drop rows with col names
                df_cohort = df_cohort.iloc[(input.num_col_lvls() - 1) :]  # drop rows with col names
                df.columns.name = 'Mutation'
                df_cohort.columns.name = 'Mutation'
                df.fillna(0, inplace=True)
                df_cohort.fillna(0, inplace=True)
                return df, df_cohort

        @reactive.calc
        def df_func():
            parsed = df_parsed_func()
            if parsed is None:
                return None
            
            df, df_cohort = parsed
            df = df.copy()
            df_cohort = df_cohort.copy()

            # Optionally remove blank samples (rows with sum <= threshold)
            try:
                if input.remove_blank():
                    thresh = float(input.blank_threshold()) if input.blank_threshold() is not None else 0.0
                    # keep rows where the sum across experiment columns is greater than threshold
                    non_blank = df.sum(axis=1) > thresh
                    df = df[non_blank]
                    df_cohort = df_cohort[df_cohort.index.isin(df.index)]
            except Exception:
                pass

            # Aggregate df_cohort by experiment prefix using selected aggregation method
            try:
                method = input.agg_method()
            except Exception:
                method = 'mean'
            if method not in ('mean', 'median', 'sum'):
                method = 'mean'
            
            grp = df_cohort.groupby(level=0, axis=1)
            if method == 'median':
                df_cohort = grp.median()
            elif method == 'sum':
                df_cohort = grp.sum()
            else:
                df_cohort = grp.mean()

            df_cohort = df_cohort.astype('float')
            df.reset_index(inplace=True)

            # Apply filtering logic
            df_meta = df_meta_func()
            if df_meta is not None and not df_meta.empty:
                if input.filter_lipid_class():
                    df = df[df["Sample Name"].isin(df_meta[df_meta["Head Group 2"].isin(input.filter_lipid_class())]["Sample Name"])]
                    df_cohort = df_cohort[df_cohort.index.isin(df["Sample Name"])]
                if input.min_chain_length():
                    df = df[df["Sample Name"].isin(df_meta[df_meta["Acyl Chain Length"] >= input.min_chain_length()]["Sample Name"])]
                    df_cohort = df_cohort[df_cohort.index.isin(df["Sample Name"])]
                if input.max_unsaturation():
                    df = df[df["Sample Name"].isin(df_meta[df_meta["Unsaturation"] <= input.max_unsaturation()]["Sample Name"])]
                    df_cohort = df_cohort[df_cohort.index.isin(df["Sample Name"])]

            if df.empty:
                return None

            return df, df_cohort

        @reactive.effect
        def update_lipid_class():
            df_meta = df_meta_func()
            if df_meta is not None and not df_meta.empty and "Head Group 2" in df_meta.columns:
                ui.update_selectize("filter_lipid_class", choices=list(df_meta["Head Group 2"].unique()))
            else:
                ui.update_selectize("filter_lipid_class", choices=[])

        @reactive.effect
        def reset_filters():
            if input.reset_filters():
                ui.update_selectize("filter_lipid_class", selected=[])
                ui.update_numeric("min_chain_length", value=0)
                ui.update_numeric("max_unsaturation", value=10)

        @reactive.calc
        def df_exps_func():
            if input.pos_data() is None or input.num_col_lvls() is None:
                return 
            else:
                if input.header() is None:
                    df_list = []
                    for file in input.pos_data():
                        chunk = pd.read_csv(file["datapath"])
                        df_list.append(chunk) 
                    df = pd.concat(df_list, ignore_index=True) # create df
                    df.drop(columns=list(df.columns[:input.num_idx()]), inplace=True) # get rid of lipid columns
                    header = df[:input.num_col_lvls()-1]
                else:
                    for h in input.header():
                        header = pd.read_csv(h['datapath'])
                h = pd.DataFrame(np.vstack([header.columns, header]))
                h.columns = [str(l) for l in range(0,len(h.columns))]
                h.index = h.index + 1
                h.reset_index(inplace=True)
                # Group header by experiment prefix
                if input.cohort_lvl():
                    w2 = int(input.cohort_lvl()) - 2
                    if w2 >= 0:
                        h.loc[w2, h.columns[1:]] = h.loc[w2, h.columns[1:]].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
                return h

        @reactive.calc
        def df_meta_func():
            parsed = df_parsed_func()
            if parsed is None:
                return None
            else:
                df_display, df = parsed
                df = df.reset_index()
                row_list = []
                for name in df["Sample Name"]:
                    # split sample name string
                    qual = re.split(' |:|;', name)
                    # get head group, chain length, unsaturation
                    head_group = qual[0]
                    # get chain length
                    if len(qual) >= 2:
                        chain_length = qual[1]
                        if "-" in chain_length:
                            c = chain_length.split(sep="-")
                            chain_length = c[1]
                            head_group += " " + c[0]
                        chain_length = int(chain_length)
                        # get unsaturation
                        if len(qual) >= 3:
                            unsaturation = qual[2]
                            if "+" in unsaturation:
                                u = unsaturation.split(sep="+")
                                unsaturation = u[0] 
                            unsaturation = int(unsaturation)
                        else:
                            unsaturation=0
                    else:
                        chain_length=0
                    # create dict for row and then add to list of rows if not already in there
                    row = {"Sample Name":name, 
                           "Head Group":head_group, 
                           "Acyl Chain Length":chain_length, 
                           "Unsaturation":unsaturation}
                    if row not in row_list:
                        row_list.append(row)
                df_meta = pd.DataFrame(row_list)

                #head group metadata - list of original head groups
                hg_list = df_meta['Head Group'].unique()
                # list of head groups metadata
                hg2_list = []
                for hg in hg_list:
                    # first sort the O groups (ex: PC, PC O)
                    if " " in hg:
                        hg2 = hg.split(" ")[0]    
                    # sort the 1/2/3 groups(GD, GT)
                    elif hg[-1] in ['1', '2', '3']:
                        hg2 = hg[:-1]
                    # get the hexcer
                    elif 'Hex' in hg:
                        hg2 = 'Hex_Cer'  
                    # all others    
                    else:
                        hg2 = hg
                    hg2_list.append(hg2)
                df_hg = pd.DataFrame({'Head Group': hg_list, 'Head Group 2': hg2_list})

                # unsaturation metadata -- merge df_meta with df_hg
                df_meta2 = df_meta.merge(df_hg, on='Head Group')
                # add unsaturation metadata
                df_meta2['Unsaturation 2'] = np.where(df_meta2['Unsaturation'] < 3, df_meta2['Unsaturation'], '>=3')

                return df_meta2

        with ui.nav_panel("Filtered Data"):
            @render.data_frame
            def render_filtered_df():
                if df_func() is None:
                    return
                df, _ = df_func()
                return df

        with ui.nav_panel("Uploaded Data"): 
            with ui.layout_sidebar():
                with ui.sidebar():
                    ui.input_numeric("num_idx", "How many columns contain lipid information?", 1)
                    ui.input_text("idx_col", "Which column contains individual lipid species?")
                    ui.input_numeric("num_col_lvls", "How many levels are there in your header?", 1)

                @render.data_frame
                def render_raw_df():
                    if input.pos_data() is None:
                        return
                    else:
                        df_list = []
                        for file in input.pos_data():
                            chunk = pd.read_csv(file['datapath'])
                            df_list.append(chunk)
                        return pd.concat(df_list, ignore_index=True)
                
        with ui.nav_panel("Column (Experiment) Metadata"):
            with ui.layout_sidebar():
                with ui.sidebar():
                    ui.input_select("cohort_lvl", "Which level of the header contains cohort names?", choices=[])
                    ui.input_select("main_col_lvl", "Which level of the header would you like to use for the column names?", choices=[])
                    ui.input_selectize("drop_col", "Select columns to drop", choices=[], multiple=True)

                @render.data_frame
                def render_df_exps():
                    return df_exps_func()

                @reactive.effect
                def choose_cohort_col():
                    df_header = df_exps_func()
                    if df_header is None:
                        return
                    else:
                        col_idx = list(df_header['index'])
                        ui.update_select('cohort_lvl', choices=col_idx)

                @reactive.effect
                def choose_main_col():
                    df_header = df_exps_func()
                    ccol = input.cohort_lvl()
                    if df_header is None or ccol is None:
                        return
                    else:
                        col_idx = list(df_header[df_header['index'] != int(ccol)]['index']) 
                        ui.update_select('main_col_lvl', choices=col_idx)

                @reactive.effect
                def drop_cols():
                    df_header = df_exps_func()
                    ccol = input.cohort_lvl()
                    mcol = input.main_col_lvl()
                    if df_header is None or mcol is None or ccol is None:
                        return
                    else:
                        df_h = df_header.set_index('index')
                        ht = df_h.T
                        # Use grouped cohort names
                        ht[int(ccol)] = ht[int(ccol)].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
                        cols = {}
                        for cohort in ht[int(ccol)].unique():
                            one = ht[ht[int(ccol)] == cohort][1]
                            mm = ht[ht[int(ccol)] == cohort][int(mcol)]
                            cols[cohort] = dict(zip(one,mm))
                        ui.update_selectize("drop_col", choices=cols)

        with ui.nav_panel("Row (Lipid) Metadata"):
            @render.data_frame
            def render_df_meta():
                if df_meta_func() is None:
                    return
                return df_meta_func()

        with ui.nav_panel("Final Dataframe"):
            @render.data_frame
            def render_df():
                if df_func() is None:
                    return
                df_display, df_cohort = df_func()
                return df_display

with ui.nav_panel("PCA"):
    @reactive.calc
    def pca_func():
        df, df_cohort = df_func()
        if df is None or df_cohort is None:
            return None, None

        '''Standardize Dataframe'''
        from sklearn.preprocessing import StandardScaler
        df_standardized = df_cohort.T
        exps = df_standardized.index

        x = df_standardized.values
        x = StandardScaler().fit_transform(x)

        '''PCA-Dataframe'''
        from sklearn.decomposition import PCA
        pca_lipids = PCA(n_components=3)
        pca = pca_lipids.fit_transform(x)
        # create dataframe with principal components
        df_pca = pd.DataFrame(pca)
        pcs = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3']
        df_pca.columns = pcs
        df_pca['Mutation'] = exps

        '''Explained Variance'''
        ev = pca_lipids.explained_variance_ratio_

        return df_pca, ev
    
    with ui.layout_columns(col_widths=(4, 8)):
        with ui.card(full_screen=True):
            ui.card_header('Explained Variance')
            @render.code
            def ev_text():
                df_pca, ev = pca_func()
                if df_pca is None:
                    return "No data available for PCA."
                return 'Explained variance per principal component: \nPC 1: {}\nPC 2: {}\nPC 3: {}'.format(ev[0], ev[1], ev[2])

            @render.plot
            def ev_graph():
                df_pca, ev = pca_func()
                if df_pca is None:
                    return
                plt.figure(figsize=(4,5))
                plt.bar(
                    x=['PC 1', 'PC 2', 'PC 3'],
                    height=ev
                )
                plt.title('Explained Variance')

        with ui.navset_card_tab():
            with ui.nav_panel('PCA - 2D'):
                @render.plot
                def pca2d():
                    df_pca, ev = pca_func()
                    if df_pca is None:
                        return
                    i = 0
                    fig, ax_nstd = plt.subplots(figsize=(5,5))
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                    for protein in df_pca['Mutation'].unique():
                        x = df_pca[df_pca['Mutation'] == protein]['Principal Component 1']
                        y = df_pca[df_pca['Mutation'] == protein]['Principal Component 2']
                        
                        ax_nstd.scatter(x, y, color=colors[i])
                        functions.confidence_ellipse(x, y, ax_nstd, n_std=3,
                                                    label=protein, alpha=0.5, facecolor=colors[i], edgecolor=colors[i], linestyle=':')
                        i += 1

                    ax_nstd.grid()
                    ax_nstd.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    ax_nstd.set_title('Principal Components Analysis with 95% Confidence Interval')
                    ax_nstd.set_xlabel('Principal Component 1 ({:.0%})'.format(ev[0]))
                    ax_nstd.set_ylabel('Principal Component 2 ({:.0%})'.format(ev[1]))

            with ui.nav_panel('PCA - 3D'):
                @render.plot
                def pca3d():
                    df_pca, ev = pca_func()
                    if df_pca is None:
                        return
                    i = 0
                    fig = plt.figure(figsize=(8,8))
                    ax_nstd = fig.add_subplot(projection='3d')
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                    for protein in df_pca['Mutation'].unique():
                        x = df_pca[df_pca['Mutation'] == protein]['Principal Component 1']
                        y = df_pca[df_pca['Mutation'] == protein]['Principal Component 2']
                        z = df_pca[df_pca['Mutation'] == protein]['Principal Component 3']
                        
                        ax_nstd.scatter(x, y, z, color=colors[i], label=protein, s=50)
                        i += 1

                    ax_nstd.grid()
                    ax_nstd.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
                    ax_nstd.set_title('3D Principal Components Analysis')
                    ax_nstd.set_xlabel('Principal Component 1 ({:.0%})'.format(ev[0]))
                    ax_nstd.set_ylabel('Principal Component 2 ({:.0%})'.format(ev[1]))
                    ax_nstd.set_zlabel('Principal Component 3 ({:.0%})'.format(ev[2]))

with ui.nav_panel("Chain Length"):
    @reactive.calc
    def df_chain_summary():
        df, df_cohort = df_func()
        if df is None or df_cohort is None:
            return None, None, None, None
        df_meta = df_meta_func()

        # try to import helper module
        try:
            from app import analysis
        except Exception:
            import analysis

        df_cl, df_cl_norm, df_cl_z, df_long = analysis.chain_length_group_tables(df_meta, df, df_cohort)
        return df_cl, df_cl_norm, df_cl_z, df_long

    with ui.layout_columns(col_widths=(6,6)):
        with ui.card(full_screen=True):
            ui.card_header("Chain Length KDE / Histogram")
            ui.input_checkbox("fit_gaussian_cl", "Fit Gaussian Curve", value=False)
            ui.input_checkbox("remove_odd_gaussian", "Remove Odd Chains for Fit", value=False)
            @render.plot
            def chain_kde_hist():
                df_cl, df_cl_norm, df_cl_z, df_long = df_chain_summary()
                if df_long is None or df_long.empty:
                    return
                try:
                    from app import analysis
                except Exception:
                    import analysis
                fig = analysis.kde_hist_plot(df_long)
                ax = fig.gca()
                
                if input.fit_gaussian_cl():
                    colors = plt.cm.tab10(np.linspace(0, 1, len(df_long['Cohort'].unique())))
                    for i, cohort in enumerate(df_long['Cohort'].unique()):
                        color = colors[i]
                        sub = df_long[df_long['Cohort'] == cohort]
                        
                        # Prepare data for fitting (group by chain length)
                        dist = sub.groupby('Acyl Chain Length')['Abundance'].sum().reset_index()
                        dist['Fraction'] = dist['Abundance'] / dist['Abundance'].sum()
                        
                        fit_data = dist
                        if input.remove_odd_gaussian():
                            fit_data = dist[dist['Acyl Chain Length'] % 2 == 0]
                        
                        popt, r2 = analysis.fit_single_gaussian(fit_data['Acyl Chain Length'].values, fit_data['Fraction'].values)
                        if popt is not None:
                            x_fit = np.linspace(dist['Acyl Chain Length'].min(), dist['Acyl Chain Length'].max(), 200)
                            y_fit = analysis.gaus(x_fit, *popt)
                            ax.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=2, label=f'Fit {cohort} (R²={r2:.2f})')
                            ax.text(0.05, 0.95 - i*0.05, f"{cohort} μ: {popt[1]:.1f}, σ: {popt[2]:.1f}", 
                                    transform=ax.transAxes, color=color, fontweight='bold', verticalalignment='top')
                    ax.legend()
                return fig

        with ui.card(full_screen=True):
            ui.card_header("Chain Length Heatmap (Z-score)")
            @render.plot
            def chain_heatmap_z():
                df_cl, df_cl_norm, df_cl_z, df_long = df_chain_summary()
                if df_cl_z is None:
                    return
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df_cl_z, cmap='coolwarm', ax=ax, annot=False)
                plt.title('Z-score Heatmap of Chain Length (per cohort)')
                plt.xlabel('Cohort')
                plt.ylabel('Acyl Chain Length')
                plt.tight_layout()
                return fig

    with ui.layout_columns(col_widths=(6,6)):
        with ui.card(full_screen=True):
            ui.card_header("Chain Length Bubble Heatmap")
            ui.input_checkbox("show_bubble_cl", "Show Bubble Heatmap", value=False)
            @render.plot
            def chain_bubble_heatmap():
                if not input.show_bubble_cl():
                    return None
                df_cl, df_cl_norm, df_cl_z, df_long = df_chain_summary()
                if df_cl_norm is None:
                    return
                # We need a clustermap result for the bubble_heatmap function
                # For simplicity, we can use the normalized data directly or perform a quick clustering
                try:
                    res = sns.clustermap(df_cl_norm, row_cluster=True, col_cluster=True)
                    plt.close() # Close the clustermap figure as we only want the result
                    # Use the existing bubble_heatmap utility from functions.py
                    # Note: bubble_heatmap returns an Altair chart, but this is a @render.plot
                    # Actually, functions.bubble_heatmap returns an Altair chart.
                    # I should check if I can render Altair in this Shiny app or if I should use a matplotlib version.
                    # Looking at functions.py, it returns (b+c).configure(background='white') which is Altair.
                    # I'll check if app.py handles Altair.
                    pass
                except Exception as e:
                    plt.figure(); plt.text(0.5,0.5,f'Clustermap failed: {e}', ha='center'); plt.axis('off'); return plt.gcf()
                
                # Check if we should use altair or matplotlib. The current render is @render.plot (matplotlib).
                # If I want to use the existing bubble_heatmap, I'd need @render_altair.
                # However, the user might prefer a matplotlib version if they are using @render.plot everywhere else.
                # Let's try to add a matplotlib-based bubble heatmap for consistency with @render.plot.
                
                fig, ax = plt.subplots(figsize=(10, 8))
                # Reorder data based on clustering
                reordered_df = functions.extract_clustered_table(res, df_cl_norm)
                
                # Plotting bubbles
                y_labels = reordered_df.index
                x_labels = reordered_df.columns
                
                X, Y = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))
                sizes = reordered_df.values.flatten() * 5000 # Scale size
                colors = reordered_df.values.flatten()
                
                scatter = ax.scatter(X.flatten(), Y.flatten(), s=sizes, c=colors, cmap='YlGnBu', alpha=0.7, edgecolors='gray')
                
                ax.set_xticks(np.arange(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
                ax.set_yticks(np.arange(len(y_labels)))
                ax.set_yticklabels(y_labels)
                
                plt.colorbar(scatter, ax=ax, label='Fraction')
                plt.title('Bubble Heatmap of Chain Length')
                plt.xlabel('Cohort')
                plt.ylabel('Acyl Chain Length')
                plt.tight_layout()
                return fig

    with ui.layout_columns(col_widths=(6,6)):
        with ui.card(full_screen=True):
            ui.card_header('Chain Length Correlation Matrix')
            @render.plot
            def chain_corr():
                df_cl, df_cl_norm, df_cl_z, df_long = df_chain_summary()
                if df_cl is None or df_cl.empty:
                    return
                df_numeric = df_cl.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
                if df_numeric.shape[1] < 2:
                    plt.figure(figsize=(6,4))
                    plt.text(0.5, 0.5, 'Not enough numeric cohorts for correlation heatmap.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                    plt.axis('off')
                    return plt.gcf()
                corr_matrix = df_numeric.T.corr()
                fig, ax = plt.subplots(figsize=(8,7))
                sns.heatmap(corr_matrix, cmap='vlag', annot=True, fmt='.2f', linewidths=.5, ax=ax, vmin=-1, vmax=1, center=0)
                plt.title('Correlation Between Chain Length Profiles')
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                return fig

    with ui.layout_columns(col_widths=(6,6)):
        with ui.card(full_screen=True):
            ui.card_header('Subsets by Chain Length')
            @render.plot
            def cl_ge_50():
                df, df_cohort = df_func()
                df_meta = df_meta_func()
                if df is None or df_meta is None:
                    return
                try:
                    from app import analysis
                except Exception:
                    import analysis
                df_hg = analysis.subset_headgroup_by_chain(df_meta, df, df_cohort, lambda x: x >= 50)
                if df_hg.empty:
                    plt.figure(); plt.text(0.5,0.5,'No lipids with chain length >= 50', ha='center'); plt.axis('off'); return plt.gcf()
                totals = df_hg.sum(axis=1)
                fig, ax = plt.subplots(figsize=(5,5))
                totals.plot(kind='bar', ax=ax)
                ax.set_title('Head group distribution for chain length >= 50')
                plt.tight_layout()
                return fig

        with ui.card(full_screen=True):
            ui.card_header('Head Groups for Chain Length <= 30')
            @render.plot
            def cl_le_30():
                df, df_cohort = df_func()
                df_meta = df_meta_func()
                if df is None or df_meta is None:
                    return
                try:
                    from app import analysis
                except Exception:
                    import analysis
                df_hg = analysis.subset_headgroup_by_chain(df_meta, df, df_cohort, lambda x: x <= 30)
                if df_hg.empty:
                    plt.figure(); plt.text(0.5,0.5,'No lipids with chain length <= 30', ha='center'); plt.axis('off'); return plt.gcf()
                totals = df_hg.sum(axis=1)
                fig, ax = plt.subplots(figsize=(5,5))
                totals.plot(kind='bar', ax=ax)
                ax.set_title('Head group distribution for chain length <= 30')
                plt.tight_layout()
                return fig

    with ui.layout_columns(col_widths=(6,6)):
        with ui.card(full_screen=True):
            ui.card_header('Head Groups for Chain Length <= 20')
            @render.plot
            def cl_le_20():
                df, df_cohort = df_func()
                df_meta = df_meta_func()
                if df is None or df_meta is None:
                    return
                try:
                    from app import analysis
                except Exception:
                    import analysis
                df_hg = analysis.subset_headgroup_by_chain(df_meta, df, df_cohort, lambda x: x <= 20)
                if df_hg.empty:
                    plt.figure(); plt.text(0.5,0.5,'No lipids with chain length <= 20', ha='center'); plt.axis('off'); return plt.gcf()
                totals = df_hg.sum(axis=1)
                fig, ax = plt.subplots(figsize=(5,5))
                totals.plot(kind='bar', ax=ax)
                ax.set_title('Head group distribution for chain length <= 20')
                plt.tight_layout()
                return fig

    with ui.card(full_screen=True):
        ui.card_header('Odd-chain fraction by Cohort')
        @render.plot
        def odd_chain_frac_plot():
            df, df_cohort = df_func()
            df_meta = df_meta_func()
            if df is None or df_meta is None:
                return
            try:
                from app import analysis
            except Exception:
                import analysis
            s = analysis.odd_chain_fraction(df_meta, df_cohort)
            if s is None or s.empty:
                plt.figure(); plt.text(0.5,0.5,'No data for odd-chain analysis', ha='center'); plt.axis('off'); return plt.gcf()
            fig, ax = plt.subplots(figsize=(8,4))
            s['FractionOdd'].plot(kind='bar', ax=ax)
            ax.set_ylabel('Fraction odd-chain')
            ax.set_title('Fraction of odd-chain lipids per cohort')
            plt.tight_layout()
            return fig

    with ui.card(full_screen=True):
        ui.card_header('Fold change vs control (Chain Length)')
        ui.input_select('control_mutation', 'Control mutation', choices=[])
        @reactive.effect
        def update_control_choices():
            df_pair = df_func()
            if df_pair is None:
                ui.update_select('control_mutation', choices=[])
                return
            df_display, df_cohort = df_pair
            if df_cohort is None:
                ui.update_select('control_mutation', choices=[])
                return
            ui.update_select('control_mutation', choices=list(df_cohort.columns))

            return plt.gcf()

    with ui.card(full_screen=True):
        ui.card_header("Significance XY Plot (Difference vs Control)")
        ui.input_selectize("sig_control_cohort", "Select Control Cohort", choices=[])
        @reactive.effect
        def update_sig_controls():
            df, df_cohort = df_func()
            if df_cohort is not None:
                ui.update_selectize("sig_control_cohort", choices=list(df_cohort.columns))
                
        @render.plot
        def sig_xy_plot():
            ctrl = input.sig_control_cohort()
            if ctrl is None:
                return
            df, df_cohort = df_func()
            df_meta = df_meta_func()
            if df is None or df_meta is None:
                return
            
            from app import analysis
            res_df = analysis.run_per_point_ttest(df_meta, df_cohort, "Acyl Chain Length", ctrl)
            if res_df.empty:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            cohorts = res_df['Cohort'].unique()
            for cohort in cohorts:
                sub = res_df[res_df['Cohort'] == cohort]
                ax.plot(sub['Level'], sub['MeanDiff'], marker='o', label=f"{cohort} vs {ctrl}", alpha=0.6)
                
                # Highlight significant points
                sig_pts = sub[sub['Significant']]
                if not sig_pts.empty:
                    ax.scatter(sig_pts['Level'], sig_pts['MeanDiff'], color='red', s=100, edgecolors='black', label=f"{cohort} Significant (p<0.05)")
            
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel("Acyl Chain Length")
            ax.set_ylabel("Mean Difference in Abundance")
            ax.set_title("Per-point T-test: Difference from Control")
            ax.legend()
            plt.tight_layout()
            return fig

with ui.nav_panel("Statistics"):
    """Run ANOVA per selected variable level and show Holm-Sidak post-hoc results."""
    with ui.layout_sidebar():
        with ui.sidebar():
            ui.input_select("stat_var", "Variable to test", choices=[], selected="Head Group 2")
            ui.input_numeric("alpha", "Alpha (significance level)", value=0.05)
            ui.input_action_button("run_stats", "Run Statistics")
            ui.input_selectize("posthoc_level", "Select level for pairwise comparisons", choices=[], multiple=False)
            @render.download(filename="posthoc_results.csv")
            def download_posthoc():
                anova_df, posthoc = stats_result()
                if anova_df is None or posthoc is None:
                    return None
                lvl = input.posthoc_level()
                if lvl is None or lvl == "":
                    return None
                if lvl not in posthoc:
                    return None
                df_ph = posthoc[lvl]
                return df_ph.to_csv(index=False)

    @reactive.effect
    def update_stat_var_choices():
        # populate variable choices from df_meta columns
        dfm = df_meta_func()
        if dfm is None:
            ui.update_select('stat_var', choices=[])
            return
        choices = [c for c in ['Head Group 2', 'Acyl Chain Length', 'Unsaturation 2'] if c in dfm.columns]
        if not choices:
            choices = list(dfm.columns)
        ui.update_select('stat_var', choices=choices, selected=('Head Group 2' if 'Head Group 2' in choices else choices[0] if choices else None))

    @reactive.calc
    def stats_result():
        # only run when user clicks the run button
        if input.run_stats() is None or input.run_stats() == 0:
            return None, None
        df_pair = df_func()
        if df_pair is None:
            return None, None
        df_display, df_cohort = df_pair
        dfm = df_meta_func()
        if dfm is None or df_cohort is None:
            return None, None
        var = input.stat_var() if input.stat_var() is not None else 'Head Group 2'
        try:
            from app import analysis
        except Exception:
            import analysis
        try:
            anova_df, posthoc = analysis.anova_per_level(dfm, df_cohort, var, alpha=float(input.alpha() if input.alpha() is not None else 0.05))
        except Exception as e:
            # return errors via an empty result and show message via text output below
            return pd.DataFrame({'Error': [str(e)]}), {}
        return anova_df, posthoc

    @reactive.effect
    def update_posthoc_levels():
        res = stats_result()
        if res is None:
            ui.update_selectize('posthoc_level', choices=[])
            return
        anova_df, posthoc = res
        if anova_df is None:
            ui.update_selectize('posthoc_level', choices=[])
            return
        levels = list(anova_df.index.astype(str))
        ui.update_selectize('posthoc_level', choices=levels)

    @render.data_frame
    def anova_table():
        res = stats_result()
        if res is None:
            return
        anova_df, _ = res
        if anova_df is None:
            return
        try:
            return anova_df.reset_index().rename(columns={'index': 'level'})
        except Exception:
            return anova_df


    @render.data_frame
    def posthoc_table():
        res = stats_result()
        if res is None:
            return
        _, posthoc = res
        lvl = input.posthoc_level()
        if lvl is None or lvl == "":
            return pd.DataFrame()
        if lvl not in posthoc:
            return pd.DataFrame()
        df_ph = posthoc[lvl]
        return df_ph

with ui.nav_panel("Head Group"):
    @reactive.calc
    def df_headgroup_summary():
        df, df_cohort = df_func()
        if df is None or df_cohort is None:
            return None, None, None, None
        df_meta = df_meta_func()

        df_merged = df_meta.merge(df, on="Sample Name")
        df_hg = df_merged.groupby("Head Group 2").sum()

        df_hg_norm = df_hg.div(df_hg.sum(axis=0), axis=1)  # normalize columns
        df_hg_z = df_hg.apply(lambda x: (x - x.mean()) / x.std(), axis=1)  # Z-score
        df_cohort = df_cohort.T.reset_index().rename(columns={"index": "Sample Name"})
        df_merged_long = df_meta.merge(df_cohort, on="Sample Name")

        df_long = df_merged_long.melt(
            id_vars=["Sample Name", "Head Group", "Head Group 2", "Acyl Chain Length", "Unsaturation", "Unsaturation 2"],
            var_name="Cohort",
            value_name="Abundance"
        )

        df_long['Z-score'] = df_long.groupby("Head Group 2")["Abundance"].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        return df_hg, df_hg_norm, df_hg_z, df_long

    with ui.layout_columns(col_widths=(6, 6)):
        with ui.card(full_screen=True):
            ui.card_header("Donut Chart (Normalized)")
            @render.plot
            def donut_chart():
                df_hg, df_hg_norm, df_hg_z, df_long = df_headgroup_summary()
                if df_hg_norm is None:
                    return

                # Sum across all samples to get total normalized abundance
                totals = df_hg_norm.sum(axis=1)
                labels = totals.index
                sizes = totals.values

                fig, ax = plt.subplots(figsize=(5, 5))
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, autopct="%1.1f%%", startangle=140
                ) 
                centre_circle = plt.Circle((0, 0), 0.70, fc="white")
                fig.gca().add_artist(centre_circle)
                ax.axis("equal")
                plt.title("Normalized Head Group Distribution")
                return fig

        with ui.card(full_screen=True):
            ui.card_header("Heatmap (Z-score of Head Groups)")
            @render.plot
            def heatmap_z():
                df_hg, df_hg_norm, df_hg_z, df_long = df_headgroup_summary()
                if df_hg_z is None:
                    return

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df_hg_z, cmap="coolwarm", ax=ax, annot=True, fmt=".2f")
                plt.title("Z-score Heatmap of Head Groups (per sample)")
                plt.xlabel("Sample")
                plt.ylabel("Head Group")
                return fig

    with ui.card(full_screen=True):
        ui.card_header("Head Group Correlation Heatmap")
        @render.plot
        def hg_correlation_heatmap():
            df_hg, df_hg_norm, df_hg_z, df_long = df_headgroup_summary()
            if df_hg is None or df_hg.empty:
                return

            df_numeric = df_hg.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

            if df_numeric.shape[1] < 2:
                plt.figure(figsize=(6,4))
                plt.text(0.5, 0.5, "Not enough numeric head groups for correlation heatmap.",
                         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.axis('off')
                return plt.gcf()

            corr_matrix = df_numeric.T.corr() 

            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(corr_matrix, cmap='vlag', annot=True, fmt=".2f",
                        linewidths=.5, linecolor='black', ax=ax, vmin=-1, vmax=1, center=0)
            plt.title("Correlation Between Head Groups")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            return fig

    with ui.card(full_screen=True):
        ui.card_header("Nightingale Rose Chart (Head Group Comparative)")
        ui.input_checkbox("show_rose", "Show Rose Chart", value=False)
        @render.plot
        def rose_chart():
            if not input.show_rose():
                return
            df_hg, df_hg_norm, df_hg_z, df_long = df_headgroup_summary()
            if df_hg_norm is None:
                return
            
            # Implementation of polar bar chart
            labels = df_hg_norm.index
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1] # close the circle
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            for cohort in df_hg_norm.columns:
                values = df_hg_norm[cohort].tolist()
                values += values[:1] # close the circle
                ax.fill(angles, values, alpha=0.25, label=cohort)
                ax.plot(angles, values, linewidth=2)
                
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
            plt.title("Head Group Distribution (Fraction)")
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            return fig

    with ui.card(full_screen=True):
        ui.card_header("Head Group Abundance by Cohort")
        ui.input_select("bar_plot_type", "Select Value Type", choices={"sum": "Total Abundance", "mean": "Average Abundance", "zscore": "Z-score"})
        ui.input_selectize("selected_head_group", "Select Head Group to Plot", choices=[])

        @reactive.effect
        def update_hg_bar_choices():
            df_hg, df_hg_norm, df_hg_z, df_merged_with_cohort = df_headgroup_summary()
            if df_merged_with_cohort is not None:
                if "Head Group 2" in df_merged_with_cohort.columns:
                    ui.update_selectize("selected_head_group", choices=list(df_merged_with_cohort["Head Group 2"].unique()))
                else:
                    ui.update_selectize("selected_head_group", choices=[], selected=None)

        @render.plot
        def hg_bar_plot_by_cohort():
            df_hg, df_hg_norm, df_hg_z, df_merged_with_cohort = df_headgroup_summary()
            plot_type = input.bar_plot_type()
            selected_hg = input.selected_head_group()

            if df_merged_with_cohort is None or selected_hg is None or 'Cohort' not in df_merged_with_cohort.columns or "Head Group 2" not in df_merged_with_cohort.columns:
                return
            
            df_filtered_hg = df_merged_with_cohort[df_merged_with_cohort['Head Group 2'] == selected_hg]
            
            if df_filtered_hg.empty:
                return
            df_filtered_hg['Z-score'] = df_filtered_hg.groupby('Cohort')["Abundance"].transform(lambda x: (x - x.mean()) / x.std())

            if plot_type == "sum":
                df_plot_data = df_filtered_hg.groupby('Cohort')['Abundance'].sum().reset_index()
                y_label = "Total Abundance"
                y_col = "Abundance"
            elif plot_type == "mean":
                df_plot_data = df_filtered_hg.groupby('Cohort')['Abundance'].mean().reset_index()
                y_label = "Average Abundance"
                y_col = "Abundance"
            elif plot_type == "zscore":
                df_plot_data = df_filtered_hg.groupby('Cohort')["Z-score"].mean().reset_index()
                y_col = "Z-score"
                y_label = "Z-score"
            else:
                return

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Cohort', y=y_col, data=df_plot_data, ax=ax, palette='viridis')
            sns.stripplot(x='Cohort', y=y_col, data=df_filtered_hg, color=".2", jitter=0.2, ax=ax)

            ax.set_title(f"{y_label} of {selected_hg} by Cohort")
            ax.set_xlabel("Cohort")
            ax.set_ylabel(y_label)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            return fig


with ui.nav_panel("Unsaturation"):
    @reactive.calc
    def df_unsaturation_summary():
        """Compute unsaturation-grouped tables."""
        df, df_cohort = df_func()
        if df is None or df_cohort is None:
            return None, None, None, None
        df_meta = df_meta_func()
        if df_meta is None:
            return None, None, None, None

        try:
            from app import analysis
        except Exception:
            import analysis

        df_unsat, df_unsat_norm, df_unsat_z, df_long = analysis.unsaturation_group_tables(df_meta, df, df_cohort)
        return df_unsat, df_unsat_norm, df_unsat_z, df_long

    with ui.layout_columns(col_widths=(6, 6)):
        with ui.card(full_screen=True):
            ui.card_header("KDE & Histogram (Unsaturation weighted by abundance)")
            @render.plot
            def kde_hist_unsat():
                df_unsat, df_unsat_norm, df_unsat_z, df_long = df_unsaturation_summary()
                if df_long is None or df_long.empty:
                    return
                try:
                    from app import analysis
                except Exception:
                    import analysis
                return analysis.kde_hist_plot_unsat(df_long, figsize=(8, 4))

        with ui.card(full_screen=True):
            ui.card_header("Heatmap (Z-score of Unsaturation Levels)")
            @render.plot
            def heatmap_unsat_z():
                df_unsat, df_unsat_norm, df_unsat_z, df_long = df_unsaturation_summary()
                if df_unsat_z is None or df_unsat_z.empty:
                    return
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df_unsat_z, cmap="coolwarm", ax=ax, annot=True, fmt=".2f")
                plt.title("Z-score Heatmap of Unsaturation (per cohort)")
                plt.xlabel("Cohort")
                plt.ylabel("Unsaturation Level (# double bonds)")
                return fig

    with ui.card(full_screen=True):
        ui.card_header("Unsaturation Subsets (Head Group Distribution)")
        with ui.layout_columns(col_widths=(4, 4, 4)):
            with ui.card():
                ui.card_header("Saturated (0 DB)")
                @render.plot
                def hg_unsat_0():
                    df_unsat, df_unsat_norm, df_unsat_z, df_long = df_unsaturation_summary()
                    if df_unsat_norm is None:
                        return
                    df_meta = df_meta_func()
                    df_cohort = df_func()[1]
                    if df_meta is None or df_cohort is None:
                        return
                    try:
                        from app import analysis
                    except Exception:
                        import analysis
                    df_subset = analysis.subset_headgroup_by_unsat(df_meta, df_cohort, lambda x: x == 0)
                    if df_subset.empty:
                        return
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(df_subset, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f")
                    plt.title("Head Groups (Saturated)")
                    return fig

            with ui.card():
                ui.card_header("Monounsaturated (1-2 DB)")
                @render.plot
                def hg_unsat_1_2():
                    df_unsat, df_unsat_norm, df_unsat_z, df_long = df_unsaturation_summary()
                    if df_unsat_norm is None:
                        return
                    df_meta = df_meta_func()
                    df_cohort = df_func()[1]
                    if df_meta is None or df_cohort is None:
                        return
                    try:
                        from app import analysis
                    except Exception:
                        import analysis
                    df_subset = analysis.subset_headgroup_by_unsat(df_meta, df_cohort, lambda x: (x >= 1) & (x <= 2))
                    if df_subset.empty:
                        return
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(df_subset, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f")
                    plt.title("Head Groups (1-2 DB)")
                    return fig

            with ui.card():
                ui.card_header("Polyunsaturated (≥3 DB)")
                @render.plot
                def hg_unsat_3_plus():
                    df_unsat, df_unsat_norm, df_unsat_z, df_long = df_unsaturation_summary()
                    if df_unsat_norm is None:
                        return
                    df_meta = df_meta_func()
                    df_cohort = df_func()[1]
                    if df_meta is None or df_cohort is None:
                        return
                    try:
                        from app import analysis
                    except Exception:
                        import analysis
                    df_subset = analysis.subset_headgroup_by_unsat(df_meta, df_cohort, lambda x: x >= 3)
                    if df_subset.empty:
                        return
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(df_subset, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f")
                    plt.title("Head Groups (≥3 DB)")
                    return fig

    with ui.card(full_screen=True):
        ui.card_header("Unsaturation Correlation Heatmap")
        @render.plot
        def unsat_correlation_heatmap():
            df_unsat, df_unsat_norm, df_unsat_z, df_long = df_unsaturation_summary()
            if df_unsat is None or df_unsat.empty:
                return
            corr = df_unsat.T.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
            plt.title("Correlation between Unsaturation Levels (across cohorts)")
            return fig

    with ui.card(full_screen=True):
        ui.card_header("Fold-Change Heatmap (Unsaturation)")
        @render.plot
        def fc_heatmap_unsat():
            df, df_cohort = df_func()
            if df is None or df_cohort is None:
                return
            df_meta = df_meta_func()
            if df_meta is None or df_meta.empty:
                return
            # Use the first (WT or control) cohort as reference
            if df_cohort.shape[1] < 2:
                return
            ctrl = df_cohort.columns[0]
            try:
                _df, non_inf_max, non_inf_min = functions.fold_change(df_meta, df, 'Unsaturation', ctrl, row_cluster=True, renamed_var='Unsaturation')
            except Exception as e:
                plt.figure()
                plt.text(0.5, 0.5, f'Fold change failed: {e}', ha='center')
                plt.axis('off')
                return plt.gcf()
            return plt.gcf()


with ui.nav_panel("Lipid Class"):
    @reactive.calc
    def df_lipidclass_summary():
        """Compute lipid class-level aggregations."""
        df, df_cohort = df_func()
        if df is None or df_cohort is None:
            return None, None, None
        df_meta = df_meta_func()
        if df_meta is None:
            return None, None, None

        try:
            from app import analysis
        except Exception:
            import analysis

        df_lc, df_lc_norm, df_lc_z = analysis.lipid_class_tables(df_meta, df_cohort)
        return df_lc, df_lc_norm, df_lc_z

    with ui.layout_columns(col_widths=(6, 6)):
        with ui.card(full_screen=True):
            ui.card_header("Pie Chart (Lipid Class Distribution)")
            @render.plot
            def pie_chart_lc():
                df_lc, df_lc_norm, df_lc_z = df_lipidclass_summary()
                if df_lc_norm is None or df_lc_norm.empty:
                    return
                # Sum across all cohorts to get total normalized abundance
                totals = df_lc_norm.sum(axis=1)
                labels = totals.index
                sizes = totals.values
                fig, ax = plt.subplots(figsize=(7, 7))
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                ax.set_title("Lipid Class Distribution (normalized)")
                return fig

        with ui.card(full_screen=True):
            ui.card_header("Heatmap (Lipid Classes Z-score)")
            @render.plot
            def heatmap_lc_z():
                df_lc, df_lc_norm, df_lc_z = df_lipidclass_summary()
                if df_lc_z is None or df_lc_z.empty:
                    return
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df_lc_z, cmap="coolwarm", ax=ax, annot=True, fmt=".2f")
                plt.title("Z-score Heatmap of Lipid Classes (per cohort)")
                plt.xlabel("Cohort")
                plt.ylabel("Lipid Class")
                return fig

    with ui.card(full_screen=True):
        ui.card_header("Normalized Heatmap (Lipid Classes)")
        @render.plot
        def heatmap_lc_norm():
            df_lc, df_lc_norm, df_lc_z = df_lipidclass_summary()
            if df_lc_norm is None or df_lc_norm.empty:
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_lc_norm, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f")
            plt.title("Normalized Heatmap of Lipid Classes (proportions)")
            plt.xlabel("Cohort")
            plt.ylabel("Lipid Class")
            return fig

    with ui.card(full_screen=True):
        ui.card_header("Fold-Change Heatmap (Lipid Class)")
        @render.plot
        def fc_heatmap_lc():
            df, df_cohort = df_func()
            if df is None or df_cohort is None:
                return
            df_meta = df_meta_func()
            if df_meta is None or df_meta.empty:
                return
            # Use the first (WT or control) cohort as reference
            if df_cohort.shape[1] < 2:
                return
            ctrl = df_cohort.columns[0]
            
            # Determine grouping variable
            group_var = 'Lipid Class' if 'Lipid Class' in df_meta.columns else 'Head Group'
            
            try:
                _df, non_inf_max, non_inf_min = functions.fold_change(df_meta, df, group_var, ctrl, row_cluster=True, renamed_var=group_var)
            except Exception as e:
                plt.figure()
                plt.text(0.5, 0.5, f'Fold change failed: {e}', ha='center')
                plt.axis('off')
                return plt.gcf()
            return plt.gcf()



with ui.nav_panel("General"):
    with ui.layout_columns(col_widths=(6, 6)):
        with ui.card(full_screen=True):
            ui.card_header("Lipidome Treemap (Global Composition)")
            ui.input_checkbox("show_treemap", "Render Treemap", value=False)
            @render.plot
            def treemap_plot():
                if not input.show_treemap():
                    return
                df, df_cohort = df_func()
                df_meta = df_meta_func()
                if df_meta is None or df_cohort is None:
                    return
                
                import plotly.express as px
                # Aggregate by Head Group 2 and Class
                df_merged = df_meta.merge(df_cohort.sum(axis=1).rename("Total"), on="Sample Name")
                # Group for hierarchy
                df_tree = df_merged.groupby(["Head Group 2", "Head Group"]).sum().reset_index()
                
                fig = px.treemap(df_tree, path=["Head Group 2", "Head Group"], values="Total",
                                 title="Lipidome Composition Hierarchy")
                
                # To render in @render.plot, we convert to static image
                from io import BytesIO
                img_bytes = fig.to_image(format="png")
                from matplotlib import image as mpimg
                img = mpimg.imread(BytesIO(img_bytes))
                
                fig_plt, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img)
                ax.axis('off')
                return fig_plt

        with ui.card(full_screen=True):
            ui.card_header("Lipidome Sunburst (Chain & Unsaturation)")
            ui.input_checkbox("show_sunburst", "Render Sunburst", value=False)
            @render.plot
            def sunburst_plot():
                if not input.show_sunburst():
                    return
                df, df_cohort = df_func()
                df_meta = df_meta_func()
                if df_meta is None or df_cohort is None:
                    return
                
                import plotly.express as px
                df_merged = df_meta.merge(df_cohort.sum(axis=1).rename("Total"), on="Sample Name")
                
                fig = px.sunburst(df_merged, path=["Acyl Chain Length", "Unsaturation"], values="Total",
                                  title="Chain Length & Unsaturation Distribution")
                
                from io import BytesIO
                img_bytes = fig.to_image(format="png")
                from matplotlib import image as mpimg
                img = mpimg.imread(BytesIO(img_bytes))
                
                fig_plt, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img)
                ax.axis('off')
                return fig_plt
    
    with ui.card(full_screen=True):
        ui.card_header("Two-Way ANOVA (Interactions)")
        ui.input_select("anova_var2", "Select Metadata Variable", choices={"Head Group 2": "Head Group", "Acyl Chain Length": "Chain Length", "Unsaturation": "Unsaturation"})
        ui.input_action_button("run_2way_anova", "Run Two-Way ANOVA")
        @render.data_frame
        def two_way_anova_table():
            if input.run_2way_anova() == 0:
                return
            df, df_cohort = df_func()
            df_meta = df_meta_func()
            if df_meta is None or df_cohort is None:
                return
            
            from app import analysis
            res = analysis.run_two_way_anova(df_meta, df_cohort, "Cohort", input.anova_var2())
            if res is not None:
                return res.reset_index()

with ui.nav_panel("Statistics"):
    with ui.card():
        ui.card_header("Dunnett's Test (Compare vs Control)")
        ui.input_select("dunnett_var", "Select Variable", choices={"Head Group 2": "Head Group", "Acyl Chain Length": "Chain Length", "Unsaturation": "Unsaturation"})
        ui.input_selectize("control_cohort", "Select Control Cohort", choices=[])
        ui.input_action_button("run_dunnett", "Run Dunnett's Test")
        
        @reactive.effect
        def update_dunnett_controls():
            df, df_cohort = df_func()
            if df_cohort is not None:
                ui.update_selectize("control_cohort", choices=list(df_cohort.columns))

        @render.data_frame
        def dunnett_table():
            if input.run_dunnett() == 0:
                return
            df, df_cohort = df_func()
            df_meta = df_meta_func()
            if df_meta is None or df_cohort is None:
                return
            
            from app import analysis
            res_dict = analysis.run_dunnett_test(df_meta, df_cohort, input.dunnett_var(), input.control_cohort())
            
            # Combine dict into one table for easy viewing
            all_dfs = []
            for lvl, df_res in res_dict.items():
                if df_res is not None:
                    df_res['Level'] = lvl
                    all_dfs.append(df_res)
            
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)

if __name__ == "__main__":
    import shiny
    shiny.run()
