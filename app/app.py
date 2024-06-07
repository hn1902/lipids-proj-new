import seaborn as sns
import pandas as pd
import re
import numpy as np

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
    ui.input_numeric("num_idx", "How many columns contain lipid information?", 1)
    ui.input_text("idx_col", "Which column contains individual lipid species?")
    ui.input_numeric("num_col_lvls", "How many levels are there in your header?", 1)
    ui.input_numeric(
        "main_col_lvl",
        "Which level of the header would you like to use for the column names?",
        1,
    )

    with ui.navset_card_underline(title="Dataframes"):

        @reactive.calc
        def df_func():
            if input.pos_data() is None:
                return
            else:
                df_list = []
                for file in input.pos_data():
                    chunk = pd.read_csv(file["datapath"])
                    df_list.append(chunk)
                df = pd.concat(df_list, ignore_index=True)
                df.set_index(input.idx_col(), inplace=True)
                df.index.name = 'Sample Name'
                df.drop(columns=list(df.columns[:input.num_idx() - 1]), inplace=True)
                w = input.main_col_lvl() - 2 # check row idx of column
                if input.header() is None: # use columns in df
                        if w >= 0: # if w < 0, use orginal column name
                            df.columns = list(df.iloc[w]) # set column names based on row idx
                else: # use submitted header
                    for h in input.header():
                        header = pd.read_csv(h['datapath'])
                        header = header.T
                    df.drop(columns=list(df.columns[~df.columns.isin(header.index)]), inplace=True) #drop any columns not in header
                    if w >= 0:
                        df.rename(columns=header[w], inplace=True) # rename based on column level
                df = df.iloc[(input.num_col_lvls() - 1) :]  # drop rows with col names
                df.columns.name = 'Mutation'
                df.fillna(0, inplace=True)
                df = df.astype('float')
                df.reset_index(inplace=True)
                return df

            
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
                return header
            
        @reactive.calc
        def df_meta_func():
            df = df_func()
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
                    chain_length = qual[1]
                    if "-" in chain_length:
                        c = chain_length.split(sep="-")
                        chain_length = c[1]
                        head_group += " " + c[0]
                    chain_length = int(chain_length)
                    # get unsaturation
                    unsaturation = qual[2]
                    if "+" in unsaturation:
                        u = unsaturation.split(sep="+")
                        unsaturation = u[0] 
                    unsaturation = int(unsaturation)
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
                    # get the acylglycerols
                    elif hg in 'DAG,TAG,MAG':
                        hg2='DAG,TAG,MAG'
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
            @render.data_frame
            def render_df_exps():
                return df_exps_func()
            
        with ui.nav_panel("Row (Lipid) Metadata"):
            @render.data_frame
            def render_df_meta():
                return df_meta_func()
            
        with ui.nav_panel("Final Dataframe"):
            @render.data_frame
            def render_df():
                return df_func()