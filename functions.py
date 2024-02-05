#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:26:39 2024

@author: madhunarendran
"""

def df_p(df, df_exps):
    '''
    Returns dataframe with columns named by mutation, rather than individual experiments. Also sets lipid samples as index
    
    Parameters
    ------------
    df: dataframe
        Original dataframe with columns that need to be renamed
    
    df_exps: dataframe
        Dataframe with experiment metadata
    '''
    df = df.rename(columns=df_exps.set_index('Exp')['Mutation'])
    df = df.set_index('Sample Name')
    df.columns.names=['Mutation']
    return df


def norm_col(df):
    '''
    Returns dataframe normalized down the columns
    
    Parameters
    -----------
    df: dataframe
        Dataframe with values for normalization. Columns must only have numerical values
    '''
    return df/df.sum()


def norm_row(df):
    '''
    Returns dataframe normalized across the row
    
    Parameters
    -----------
    df: dataframe
        Dataframe with values for normalization. Rows must only have numerical values
    '''
    return df.div(df.sum(axis=1), axis=0)


def groupby_norm(df_meta, df_p, var, renamed_var='', drop_var=[], drop_mutation=[], norm_exp=True, norm_var=True):
    '''
    Returns the groupby for a dataset when comparing by specific lipid qualities (ex: chain length, head group, unsaturation).
    Data is normalized by column and row. 

    Parameters
    -----------
    df_meta: dataframe
        Dataframe with lipid metadata (sample name, head group, chain length, unsaturation)
    df_p: dataframe
        Dataframe with columns named by mutation, rather than individual experiments
    var: str
        Column name for variable of interest (ex: 'Head Group 2')
    renamed_var: str, optional, default '""'
        Rename variable column to this string (ex: if var is 'Head Group 2', renamed_var is 'Head Group')
    drop_var: list, optional, default '[]'
        Variables (rows) to exclude (ex: PE, PC). 
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row).
    drop_mutation: list, optional, default '[]'
        Mutations (columns) to be dropped (ex: WT). Pass list of column names to drop.
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row).
    norm_exp: bool, optional, deafult 'True'
        Whether or not to normalize by mutation (down the column).
    norm_var: bool, optional, default 'True'
        Whether or not to normalize by variable (across the row).
    '''
    
    # create merged df with variable of interest
    df = df_meta[['Sample Name', var]].merge(df_p, on='Sample Name')
    
    # group by variable of interest
    df = df.groupby(var).sum()

    # normalize by the experiment (down the columns)
    if norm_exp:
        df = norm_col(df)
        
    # drop rows and columns
    if drop_var != []:
        df = df[~df.index.isin(drop_var)]
    if drop_mutation != []:
        df = df.drop(columns=drop_mutation)
    
    # name columns so that we can group by them once transposed
    df.columns.names = ['Mutation']
    # rename index
    if renamed_var != '':
        df.index.names = ['renamed_var']
        
    # transpose and then find the average value for each mutation
    df = df.T.groupby('Mutation').mean()
    
    # transpose again so variable is in the row and mutation is in the columns
    df = df.T
    # normalize by the variable
    if norm_var:
        df = norm_row(df)
        
    return df


def norm_long(df_meta, df_p, var, renamed_var='', drop_var=[], drop_mutation=[], norm_exp=True, norm_var=False):
    '''
    Returns the long form of a dataset when comparing by specific lipid qualities. Data is normalized by column, by default.
    Can use this to create altair charts with error bars

    Parameters
    ----------
    df_meta : dataframe
        Dataframe with lipid metadata (sample name, head group, chain length, unsaturation)
    df_p : dataframe
        Dataframe with columns named by mutation, rather than individual experiments
    var : str
        Column name for variable of interest (ex: 'Head Group 2')
    renamed_var : str, optional
        Rename variable column to this string (ex: if var is 'Head Group 2', renamed_var is 'Head Group'). The default is ''.
    drop_var : list, optional
        Variables (rows) to exclude (ex: PE, PC). 
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row).
        The default is [].
    drop_mutation : list, optional
        Mutations (columns) to be dropped (ex: WT). Pass list of column names to drop.
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row). 
        The default is [].
    norm_exp : bool, optional
        Whether or not to normalize by mutation (down the column). The default is True.
    norm_var : bool, optional
        Whether or not to normalize by variable (across the row). The default is False.

    Returns
    -------
    df_long : dataframe
        Long form of dataframe grouped by variable. Column names are ['Mutation', renamed_var OR var, 'Fraction']

    '''
    # merge raw data wiht lipid metadata
    df = df_meta[['Sample Name', var]].merge(df_p, on='Sample Name')
    
    # group by variable
    df = df.groupby(var).sum()
    
    # normalize by the experiment (down the columns)
    if norm_exp:
        df = norm_col(df)
        
    # drop rows and columns
    if drop_var != []:
        df = df[~df.index.isin(drop_var)]
    if drop_mutation != []:
        df = df.drop(columns=drop_mutation)
        
    # normalize by the variable (across the row)
    if norm_var:
        df = norm_row(df)
        
    # name columns so that we can group by them once transposed
    df.columns.names = ['Mutation']
    # rename variable column name
    if renamed_var != '':
        var = renamed_var
        
    # convert to long form
    df_long = df.T.reset_index().melt('Mutation', var_name=var, value_name='Fraction')
    
    return df_long
     

def extract_clustered_table(res, data):
    """
    input
    =====
    res:     <sns.matrix.ClusterGrid>  the clustermap object
    data:    <pd.DataFrame>            input table
    
    output
    ======
    returns: <pd.DataFrame>            reordered input table
    """
    
    # if sns.clustermap is run with row_cluster=False:
    if res.dendrogram_col is None:
        print("Apparently, columns were not clustered.")
        return -1
    
    if res.dendrogram_row is not None:
        # reordering index and columns
        new_cols = data.columns[res.dendrogram_col.reordered_ind]
        new_ind = data.index[res.dendrogram_row.reordered_ind]
        
        return data.loc[new_ind, new_cols]
    
    else:
        # reordering the columns
        new_cols = data.columns[res.dendrogram_col.reordered_ind]

        return data.loc[:,new_cols]
    
def bubble_heatmap(res, data, var_type, cmap='yellowgreenblue', cmap_domain=[]):
    '''
    Creates a heatmap with bubbles depicting population size.

    Parameters
    ----------
    res : sns.matrix.ClusterGrid
        Seaborn clustermap object.
    data : pd.DataFrame
        Input table.
    var_type : str
        Variable type for y. Options are 'quantitative', 'ordinal', 'nominal',.
    cmap : str, optional
        Altair color scheme. The default is 'yellowgreenblue'.
    cmap_domain : list with 2 values, optional
         Max and min values for colormap. The default is []. If not inputted, will default to min and max values of data.

    Returns
    -------
    altair.LayerChart
        Altair heatmap with bubbles depicting population size.

    '''
    import altair as alt 
    
    # get ordered table
    res_ordered = extract_clustered_table(res, data)
    
    # get variable name and axis title
    var = res_ordered.index.name
    var_title = var.rsplit(' ', 1)[0]
    
    # create long form of data
    l = res_ordered.reset_index().melt(var, value_name='Fraction')
    
    # get domain for cmap
    if cmap_domain == []:
        cmap_domain = [min(l['Fraction']), max(l['Fraction'])]
    
    # create grid for heatmap
    b = alt.Chart(l).mark_rect(stroke='lightgray', fill=None).encode(
        x = alt.X('Mutation:N', sort=None),
        y = alt.Y(var, type=var_type, sort=None)
    ).properties(
        width=800,
        height= ((800/len(res_ordered.columns)) * len(res_ordered.index)),
        title= 'Heatmap of {var_name}, Normalized by Mutations and {var_name}'.format(var_name=var_title)   
    )
      
    # create bubbles for heatmap
    c = alt.Chart(l).mark_circle().encode(
        x = alt.X('Mutation:N', sort=None),
        y = alt.Y(var, type=var_type, sort=None).title(var_title),
        size=alt.Size('Fraction:Q').scale(domain=cmap_domain, range=[1000,6500]),
        color=alt.Color('Fraction:Q').scale(scheme=cmap, domain=cmap_domain),
        tooltip=['Mutation', var, 'Fraction']
    ) 
    
    return (b+c).configure(background='white')
    

    
    
    
    