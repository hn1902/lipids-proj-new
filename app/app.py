import seaborn as sns
import pandas as pd

# Import data from shared.py
from shared import df

from shiny.express import input, render, ui

ui.page_opts(title="Shiny navigation components")

ui.nav_spacer()  # Push the navbar items to the right

footer = ui.input_select(
    "var", "Select variable", choices=["bill_length_mm", "body_mass_g"]
)

with ui.nav_panel("Page 1"):
    with ui.navset_card_underline(title="Penguins data", footer=footer):
        with ui.nav_panel("Plot"):

            @render.plot
            def hist():
                p = sns.histplot(
                    df, x=input.var(), facecolor="#007bc2", edgecolor="white"
                )
                return p.set(xlabel=None)

        with ui.nav_panel("Table"):

            @render.data_frame
            def data():
                return df[["species", "island", input.var()]]


with ui.nav_panel("Page 2"):
    ui.input_file('pos_data', 'Select file', accept=['.csv'], multiple=True)

    @render.data_frame
    def pos_dataframe():
        df_list = []
        
        for file in input.pos_data():
            chunk = pd.read_csv(file['datapath'])
            
            cols = {}    # rename df columns and create metadata
            for name in chunk.columns[1:]:
                n = name.split('L-')[1]    # remove 'PosMSMSALL/NegMSMSALL'
                cols[name] = n
            chunk = chunk.rename(columns=cols)    # rename df columns and create metadata
            
            df_list.append(chunk)
        df = pd.concat(df_list, ignore_index=True)
        return df