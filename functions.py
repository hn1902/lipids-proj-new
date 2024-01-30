#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:26:39 2024

@author: madhunarendran
"""

'''
Returns dataframe with columns named by mutation, rather than individual experiments. Also sets lipid samples as index
Input:
    df: original dataframe with columns that need to be renamed
    df_exps: dataframe with experiment metadata
'''
def df_p(df, df_exps):
    df = df.rename(columns=df_exps.set_index('Exp')['Mutation'])
    df = df.set_index('Sample Name')
    df.columns.names=['Mutation']
    return df


'''
Input: dataframe
Returns dataframe normalized down the columns
'''
def norm_col(df):
    return df/df.sum()


'''
Returns dataframe normalized across the row
Input: dataframe
'''
def norm_row(df):
    return df.div(df.sum(axis=1), axis=0)


'''
Returns the groupby for a dataset when comparing by specific lipid qualities (ex: chain length, head group, unsaturation).
Data is normalized by column and row. 

Inputs:
df_meta: df with lipid metadata (sample, head group, chain length, unsaturation)
df_p: df with columns named by mutation, rather than individual experiments
col: column name for variable of interest (ex: head group 2)
drop_var: variable of interest (row) to be dropped (ex: head group PE)
drop_protein: mutation (column) to be dropped (ex: WT)
norm_exp: normalize by mutation (down the column), accepts boolean
norm_var: normalize by variable (across the row), accepts boolean
'''
def groupby_norm(df_meta, df_p, var, drop_var=[], drop_mutation=[], norm_exp=True, norm_var=True):
    # create merged df with variable of interest
    df = df_meta[['Sample Name', var]].merge(df_p, on='Sample Name')
    
    # group by variable of interest
    df = df.groupby(var).sum()
    # normalize by the experiment (down the columns)
    if norm_exp:
        df = norm_col(df)
        
    # drop rows and columns
    if drop_var != []:
        df = df.drop(drop_var)
    if drop_mutation != []:
        df = df.drop(columns=drop_mutation)
    
    # name columns so that we can group by them once transposed
    df.columns.names=['Mutation']
    # transpose and then find the average value for each mutation
    df = df.T.groupby('Mutation').mean()
    
    # transpose again so variable is in the row and mutation is in the columns
    df = df.T
    # normalize by the variable
    if norm_var:
        df = norm_row(df)
        
    return df