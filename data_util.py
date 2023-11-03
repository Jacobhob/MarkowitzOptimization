"""
data_util.py
Author: Jacob Lu
Date: 2022.8.28
"""

import numpy as np
import pandas as pd

np.random.seed(1)
RISK_FREE = 0.03/250

def get_csi300_components() -> pd.DataFrame:
    df = pd.read_csv('CSI300_comp.csv', index_col=0).set_index('Ticker').sort_index()
    return df

def get_csi300_hist() -> pd.DataFrame:
    df = pd.read_csv('CSI300_prc.csv', index_col=0).set_index('Date').sort_index(axis=1)
    return df

def get_csi300_alpha() -> pd.Series:
    prc = get_csi300_hist()
    ret = (prc - prc.shift(1)) / prc.shift(1)
    alpha = ret.mean() + pd.Series(np.random.rand(len(ret.columns)), index=ret.columns) * RISK_FREE
    return alpha

def get_csi300_risk() -> pd.Series:
    prc = get_csi300_hist()
    ret = (prc - prc.shift(1)) / prc.shift(1)
    var = ret.std()
    return var

def get_csi300_covariance() -> pd.DataFrame:
    prc = get_csi300_hist()
    ret = (prc - prc.shift(1)) / prc.shift(1)
    ret = ret.dropna()
    cov = ret.cov()
    return cov
