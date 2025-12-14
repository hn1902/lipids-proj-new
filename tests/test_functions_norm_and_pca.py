import pandas as pd
import numpy as np
from functions import norm_col, norm_row
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def test_norm_col_and_row():
    df = pd.DataFrame({
        'a':[1,2,3],
        'b':[2,2,2]
    })
    nc = norm_col(df.copy())
    # columns should sum to 1
    assert np.allclose(nc.sum(axis=0).values, [1.0,1.0])
    nr = norm_row(df.copy())
    # rows sum to 1
    assert np.allclose(nr.sum(axis=1).values, [1.0,1.0,1.0])


def test_pca_smoke():
    X = np.array([[1.0,2.0,3.0],[2.0,3.0,4.0],[1.0,0.0,1.0]])
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)
    assert Z.shape == (3,2)
