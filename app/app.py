import seaborn as sns
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import functions

from shiny import reactive
from shiny.express import input, render, ui

ui.page_opts(title="Lipidomics Analysis Pipeline")

ui.nav_spacer()  # Push the navbar items to the right

# footer = ui.input_select(
#     "var", "Select variable", choices=["bill_length_mm", "body_mass_g"]
# )

# with ui.nav_panel("Page 1"):
#     with ui.navset_card_underline(title="Penguins data", footer=footer):
#         with ui.nav_panel("Plot"):

#             @render.plot
#             def hist():
#                 p = sns.histplot(
#                     df, x=input.var(), facecolor="#007bc2", edgecolor="white"
#                 )
#                 return p.set(xlabel=None)

#         with ui.nav_panel("Table"):

#             @render.data_frame
#             def data():
#                 return df[["species", "island", input.var()]]


with ui.nav_panel("Upload Data"):
    ui.input_file("pos_data", "Select data files", accept=[".csv"], multiple=True)
    ui.input_file(
        "header", "Select header file (optional)", accept=[".csv"], multiple=False
    )
    # ui.input_file(
    #     "lipid_data",
    #     "Select lipid metadata file (optional)",
    #     accept=[".csv"],
    #     multiple=False,
    # )

    with ui.navset_card_underline(title="Dataframes"):   
        @reactive.calc
        def df_func():
            if input.pos_data() is None or input.main_col_lvl() is None or input.cohort_lvl() is None or input.drop_col() is None:
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
                df = df[df.columns[~df.columns.isin(input.drop_col())]]
                w = int(input.main_col_lvl()) - 2 # check row idx of column
                w2 = int(input.cohort_lvl()) - 2 # check row_idx of cohort column
                df_cohort = df.copy()
                if input.header() is None: # use columns in df
                        if w >= 0: # if w < 0, use orginal column name
                            df.columns = list(df.iloc[w]) # set column names based on row idx
                        if w2 >= 0:
                            df_cohort.columns = list(df_cohort.iloc[w2]) # set cohort columns
                else: # use submitted header
                    for h in input.header():
                        header = pd.read_csv(h['datapath'])
                        header = header.T
                    df.drop(columns=list(df.columns[~df.columns.isin(header.index)]), inplace=True) #drop any columns not in header
                    df_cohort.drop(columns=list(df_cohort.columns[~df_cohort.columns.isin(header.index)]), inplace=True) #drop any columns not in header
                    if w >= 0:
                        df.rename(columns=header[w], inplace=True) # rename based on column level
                    if w2 >= 0:
                        df_cohort.rename(columns=header[w2], inplace=True) # rename based on column level
                df = df.iloc[(input.num_col_lvls() - 1) :]  # drop rows with col names
                df_cohort = df_cohort.iloc[(input.num_col_lvls() - 1) :]  # drop rows with col names
                df.columns.name = 'Mutation'
                df_cohort.columns.name = 'Mutation'
                df.fillna(0, inplace=True)
                df_cohort.fillna(0, inplace=True)
                df_cohort = df_cohort.astype('float')
                df.reset_index(inplace=True)
                return df, df_cohort

            
        @reactive.calc
        def df_exps_func():
            if input.pos_data() is None:
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
                return h
            
        @reactive.calc
        def df_meta_func():
            df_display, df = df_func()
            df = df.reset_index()
            if df is None:
                return
            else:
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
                    ui.input_select("main_col_lvl", "Which level of the header would you like to use for the column names?",choices=[])
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
                    ccol = input.cohort_lvl() # select returns string, so we convert back to int
                    mcol = input.main_col_lvl()
                    if df_header is None or mcol is None or ccol is None:
                        return
                    else:
                        df_h = df_header.set_index('index')
                        ht = df_h.T
                        cols = {}
                        for cohort in ht[int(ccol)].unique():
                            one = ht[ht[int(ccol)] == cohort][1]
                            mm = ht[ht[int(ccol)] == cohort][int(mcol)]
                            cols[cohort] = dict(zip(one,mm))
                        ui.update_selectize("drop_col", choices=cols)

            
        with ui.nav_panel("Row (Lipid) Metadata"):
            @render.data_frame
            def render_df_meta():
                return df_meta_func()
            
        with ui.nav_panel("Final Dataframe"):
            @render.data_frame
            def render_df():
                df_display, df_cohort = df_func()
                return df_display
            
with ui.nav_panel("PCA"):
    @reactive.calc
    def pca_func():
        df, df_cohort = df_func()

        '''Standardize Dataframe'''
        from sklearn.preprocessing import StandardScaler
        df_standardized = df_cohort.T
        exps = df_standardized.index

        x = df_standardized.values
        print(x.shape)
        x = StandardScaler().fit_transform(x)

        '''PCA-Dataframe'''
        from sklearn.decomposition import PCA
        pca_lipids = PCA(n_components=3)
        pca=pca_lipids.fit_transform(x)
        # create dataframe with principal components
        df_pca = pd.DataFrame(pca)
        pcs = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3']
        df_pca.columns=pcs
        df_pca['Mutation'] = exps

        '''Explained Variance'''
        ev = pca_lipids.explained_variance_ratio_

        return df_pca, ev

    with ui.layout_columns():
        with ui.card():
            ui.card_header('Explained Variance')
            @render.code
            def ev_text():
                df_pca, ev = pca_func()
                return 'Explained variance per principal component:\nPC 1: {}\nPC 2: {}\nPC 3: {}'.format(ev[0],ev[1],ev[2])

            @render.plot
            def ev_graph():
                df_pca, ev = pca_func()
                plt.figure(figsize=(4,5))
                plt.bar(
                    x=['PC 1', 'PC 2', 'PC 3'],
                    height=ev
                )
                plt.title('Explained Variance')
        
        with ui.card():
            ui.card_header('PCA - 2D')
            @render.plot
            def pca2():
                df_pca, ev = pca_func()
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
                #plt.show()







with ui.nav_panel("Head Group"):
    "Donut chart (Normalized) + Heatmap (Normalized) + Z-Score\nadd option to drop head groups"