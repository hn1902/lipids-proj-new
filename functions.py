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
    df = df_meta[['Sample Name', var]].merge(df_p, on='Sample Name').set_index('Sample Name')
    
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
        df.index.names = [renamed_var]
        
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
    df = df_meta[['Sample Name', var]].merge(df_p, on='Sample Name').set_index('Sample Name')
    
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
    
def bubble_heatmap(res, data, var_type, title, cmap='yellowgreenblue', cmap_domain=[], domain_mid=0):
    '''
    Creates a heatmap with bubbles depicting population size.

    Parameters
    ----------
    res : sns.matrix.ClusterGrid
        Seaborn clustermap object.
    data : pd.DataFrame
        Input table.
    var_type : str
        Variable type for y. Options are 'quantitative', 'ordinal', 'nominal'.
    title : str
        Chart title
    cmap : str, optional
        Altair color scheme. The default is 'yellowgreenblue'.
    cmap_domain : list with 2 values, optional
        Max and min values for colormap. The default is []. If not inputted, will default to min and max values of data.
    domain_mid : int
        Midpoint of colormap

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
        title=title  
    )
      
    # create bubbles for heatmap
    c = alt.Chart(l).mark_circle().encode(
        x = alt.X('Mutation:N', sort=None),
        y = alt.Y(var, type=var_type, sort=None).title(var_title),
        size=alt.Size('Fraction:Q').scale(domain=cmap_domain, domainMid=domain_mid, range=[1000,6500]),
        color=alt.Color('Fraction:Q').scale(scheme=cmap, domain=cmap_domain, domainMid=domain_mid),
        tooltip=['Mutation', var, 'Fraction']
    ) 
    
    return (b+c).configure(background='white')
    

def fold_change(df_meta, df_p, var, mtn, row_cluster, renamed_var='', drop_var=[], drop_mutation=[], outlier=True, norm_exp=True, cbar_args={}):
    '''
    Creates a seaborn heatmap of the log(fold change), with outliers. Also returns a dataframe of the log(fold change)
    Data is normalized by column.

    Parameters
    -----------
    df_meta: dataframe
        Dataframe with lipid metadata (sample name, head group, chain length, unsaturation)
    df_p: dataframe
        Dataframe with columns named by mutation, rather than individual experiments
    var: str
        Column name for variable of interest (ex: 'Head Group 2')
    mtn: str
        Column name for mutation to set at 0. Normalize other mutations relative to this column.
    row_cluster : boolean
        If True, will cluster rows in heatmap.
    renamed_var: str, optional, default '""'
        Rename variable column to this string (ex: if var is 'Head Group 2', renamed_var is 'Head Group')
    drop_var: list, optional, default '[]'
        Variables (rows) to exclude (ex: PE, PC). 
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row).
    drop_mutation: list, optional, default '[]'
        Mutations (columns) to be dropped (ex: WT). Pass list of column names to drop.
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row).
    norm_exp: bool, optional, default 'True'
        Whether or not to normalize by experiment (down the column).
    outlier : boolean
        Whether to replace inf with outliers or with max/min values
        
    Returns
    -------
    df: pandas.DataFrame
        Dataframe of log(fold change) values.
    '''
    
    # import modules
    import numpy as np
    import matplotlib
    import seaborn as sns
    
    # create merged df with variable of interest
    df = df_meta[['Sample Name', var]].merge(df_p, on='Sample Name').set_index('Sample Name')
    
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
        df.index.names = [renamed_var]
        var = renamed_var
        
    # transpose and then find the average value for each mutation
    df = df.T.groupby('Mutation').mean()
    
    # transpose again so variable is in the row and mutation is in the columns
    df = df.T
    
    # get fold change
    df = df.div(df[mtn], axis=0)    # calculate fold change
    df.fillna(1, inplace=True)    # replace NaN (0/0) values with 1
    non_inf_max = max(df[df != np.inf].max())    # find max fold chage to replace infinities
    if outlier:
        df.replace(np.inf, non_inf_max **10, inplace=True)    # replace pos infinities with obvious outlier
    else:
        df.replace(np.inf, non_inf_max, inplace=True)    # replace pos infinities with max

    
    # get log of fold change
    df = np.log(df)
    non_inf_min = min(df[df != -np.inf].min())    # find minimum fold change
    non_inf_max=np.log(non_inf_max)    # change non_inf_max to log form
    if outlier:
        df.replace(-np.inf, non_inf_min*10, inplace=True)    # replace neg inf with obvious outlier
    else:
        df.replace(-np.inf, non_inf_min, inplace=True)    # replace neg inf with minimum
    
    # plot heatmap -- get colormap
    cmap = matplotlib.colormaps['RdBu']    # set colormap
    cmap.set_over('green')    # add color for pos outliers
    cmap.set_under('yellow')    # add color for neg outliers
    
    # get args for colorbar
    cbar = {}
    cbar['vmax'] = non_inf_max
    cbar['vmin'] = non_inf_min
    cbar['tmin'] = truncate(cbar['vmin'])
    cbar['tmax'] = truncate(cbar['vmax'])
    
    # replace with manual oversets if given
    for key in cbar_args.keys():
        cbar[key] = cbar_args[key]
    
    # set cbar args
    vmin=cbar['vmin']
    vmax=cbar['vmax']
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)    # set center of colorbar at 0
    ticks = [cbar['tmin'], cbar['tmin']/2, 0, cbar['tmax']/2, cbar['tmax']]    # set colorbar ticks
    cbar_kws = {'ticks' : ticks}
    if outlier:
        cbar_kws['extend'] = 'both'
    
    # plot heatmap
    sns.set(sns.set(rc={"figure.facecolor": "white"}))    # add background color for graph
    sns.clustermap(
        df, 
        cmap=cmap,
        norm=norm,
        cbar=True, cbar_kws=cbar_kws, 
        vmin=vmin, 
        vmax=vmax, 
        row_cluster=row_cluster, 
    ).fig.suptitle(
            'Heatmap of {var_name}, log(Fold Change from CAS9)'.format(var_name=var), 
            y=1.05
    )
                                                   
    return df, non_inf_max, non_inf_min

def truncate(x):
    import math
    dec = abs(math.floor(math.log10(abs(x))))
    return int(x * 10**dec) / 10**dec

def z_score(df_meta, df_p, var, ctrl, row_cluster, renamed_var='', drop_var=[], drop_mutation=[], outlier=True, norm_exp=True, cbar_args={}):
    '''
    Creates a seaborn heatmap of the z-score, with outliers. Also returns a dataframe of the z-scores
    Data is normalized by column.

    Parameters
    -----------
    df_meta: dataframe
        Dataframe with lipid metadata (sample name, head group, chain length, unsaturation)
    df_p: dataframe
        Dataframe with columns named by mutation, rather than individual experiments
    var: str
        Column name for variable of interest (ex: 'Head Group 2')
    mtn: str
        Column name for mutation to set at 0. Normalize other mutations relative to this column.
    row_cluster : boolean
        If True, will cluster rows in heatmap.
    renamed_var: str, optional, default '""'
        Rename variable column to this string (ex: if var is 'Head Group 2', renamed_var is 'Head Group')
    drop_var: list, optional, default '[]'
        Variables (rows) to exclude (ex: PE, PC). 
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row).
    drop_mutation: list, optional, default '[]'
        Mutations (columns) to be dropped (ex: WT). Pass list of column names to drop.
        Will be dropped after normalization by mutation (down column) but before normalization by variable (across row).
    norm_exp: bool, optional, default 'True'
        Whether or not to normalize by experiment (down the column).
    outlier : boolean
        Whether to replace inf with outliers or with max/min values
        
    Returns
    -------
    dfz: pandas.DataFrame
        Dataframe of z-scores.
    '''
    # import modules
    import numpy as np
    import matplotlib
    import seaborn as sns

    # create merged df with variable of interest
    df = df_meta[['Sample Name', var]].merge(df_p, on='Sample Name').set_index('Sample Name')

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
        df.index.names = [renamed_var]
        var = renamed_var

    # transpose and then find the average value & standard error 
    dfm = df.T.groupby('Mutation').mean().T
    dfm.head()

    # find standard error for each mutation
    dfse = df.T.groupby('Mutation').sem().T
    dfse.head()

    # calculate z-score
    dfm = dfm.sub(dfm[ctrl], axis=0) # numerator
    dfse = dfse ** 2 # denominator
    dfse = np.sqrt(dfse.add(dfse[ctrl], axis=0))
    dfz = dfm/dfse

    # dfz formatting
    dfz.fillna(0, inplace=True) # replace NaN (0/0) with 1
    non_inf_max = max(dfz[dfz != np.inf].max())    # find max fold chage to replace infinities
    non_inf_min = min(dfz[dfz != -np.inf].min())    # find minimum
    if outlier:
        dfz.replace(np.inf, non_inf_max **10, inplace=True)    # replace pos infinities with obvious outlier
        dfz.replace(-np.inf, non_inf_min*10, inplace=True)    # replace neg inf with obvious outlier
    else:
        dfz.replace(np.inf, non_inf_max, inplace=True)    # replace pos infinities with max
        dfz.replace(-np.inf, non_inf_min, inplace=True)    # replace neg inf with minimum

    # plot heatmap -- get colormap
    if ((-2 <= dfz) & (dfz <= 2)).all().all():
        cmap = matplotlib.colormaps['PRGn']

        # set cbar args
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)    # set center of colorbar at 0
        ticks = [-2, -1, 0, 1, 2]    # set colorbar ticks
        cbar_kws = {'ticks' : ticks}

    else:
        levels = [-30, -20, -10, -2, 2, 10, 20, 30]
        if outlier:
            colors = sns.color_palette('PRGn', 9)
            cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend='both')
        else:
            colors = sns.color_palette('PRGn', 7) 
            cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
        cbar_kws = {'ticks' : levels}

    # plot heatmap
    sns.set(sns.set(rc={"figure.facecolor": "white"}))    # add background color for graph
    sns.clustermap(
        dfz, 
        cmap=cmap,
        norm=norm,
        cbar=True, 
        cbar_kws=cbar_kws, 
        row_cluster=row_cluster, 
    ).fig.suptitle(
            '{var_name} Two-Sample Z-Test Compared to {control}'.format(var_name=var, control=ctrl), 
            y=1.05
    )

    return dfz
    
def altair_heatmap(dfz, var, var_type, val, title='', subtitle='', cmap='purplegreen', cmap_mid=0):
    '''
    Creates an altair heatmap from a fold change or z-score dataframe.

    Parameters
    -----------
    dfz : dataframe
        Dataframe with z-score or fold change data
    var : str
        Column name for variable of interest
    var_type : str
        Variable type for y. Options are 'quantitative', 'ordinal', 'nominal'.
    val: str
        Name for value of interest (ex: Z-Score, log(Fold Change))
    title : str
        Chart title
    subtitle: str, optional
        Chart subtitle
    cmap : str, optional
        Altair color scheme. The default is 'purplegreen'.
    cmap_mid : int
        Midpoint of colormap
    '''
    import altair as alt

    source = dfz.reset_index().melt(var, value_name=val)

    alt.Chart(source).mark_rect().encode(
        x='Mutation:N',
        y=alt.Y(var, type=var_type),
        color=alt.Color(val, type='quantitative').scale(scheme=cmap, domainMid=cmap_mid),
        tooltip=['Mutation', var, 'Z-Score']
    ).properties(
        height=500,
        width=500,
        title={'text':title, 'subtitle':subtitle}
    )