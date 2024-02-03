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
     
     
    
    

    
    
    
    