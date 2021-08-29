import typing as t
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import MinMaxScaler
import gower
import itertools
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

def precompute_fx(X: np.ndarray, Y: np.ndarray) -> t.Dict[str, t.Any]:

    prepcomp_vals = {}
    
    classes, class_freqs = np.unique(Y, return_counts=True)
    cls_index = [np.equal(Y, i) for i in range(classes.shape[0])]

    #cls_n_ex = np.array([np.sum(aux) for aux in cls_index])
    cls_n_ex = list(class_freqs)
    ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))
    prepcomp_vals["ovo_comb"] = ovo_comb
    prepcomp_vals["cls_index"] = cls_index
    prepcomp_vals["cls_n_ex"] = cls_n_ex
    return prepcomp_vals

"""## 2.3 Neighborhood Measures

### 2.3.1 Fraction of Borderline Points (N1)
"""

def ft_N1(X: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_ = scaler.transform(X)
    dist_m = gower.gower_matrix(X)
    mst = minimum_spanning_tree(dist_m)
    node_i, node_j = np.where(mst.toarray() > 0)
    which_have_diff_cls = y[node_i] != y[node_j]
    aux = np.unique(np.concatenate([node_i[which_have_diff_cls],node_j[which_have_diff_cls]])).shape[0]
    return aux/X.shape[0]


"""### 2.3.2 Ratio of Intra/Extra Class Nearest Neighbor Distance (N2)"""

def inter(X, Y, dst, i):
    precomp = precompute_fx(X, Y)
    cls_index_ = precomp['cls_index']
    np.logical_not(cls_index_[Y[i]])
    return min(dst[i, np.logical_not(cls_index_[Y[i]])])

def intra(X, Y, dst, i):
    precomp = precompute_fx(X, Y)
    cls_index_ = precomp['cls_index']
    cls_index_[Y[i]][i] = False
    return min(dst[i,cls_index_[Y[i]]])

def ft_N2 (X: np.ndarray, Y: np.ndarray) -> float:
    inter_ = np.zeros(X.shape[0])
    intra_ = np.zeros(X.shape[0])
    dst = gower.gower_matrix(X)
    for i in range(X.shape[0]):
        inter_[i] = inter(X, Y, dst, i)
        intra_[i] = intra(X, Y, dst, i)
    intra_inter = intra_.sum()/inter_.sum()
    return intra_inter/(1+intra_inter)


"""### 2.3.3 Error Rate of the Nearest Neighbor Classifier (N3)"""

def ft_N3 (X: np.ndarray, Y: np.ndarray, metric: str = "euclidean") -> float:
    
    dst = np.asarray(gower.gower_matrix(X))
    loo = LeaveOneOut()
    loo.get_n_splits(X, Y)

    y_test_ = []
    pred_y_ = []
    for train_index, test_index in loo.split(X):
        dst[test_index, test_index] = 100
        fnn_index = np.argmin(dst[test_index])
        y_test_.append(Y[test_index])
        pred_y_.append(Y[fnn_index])
   
    error = 1 - accuracy_score(y_test_, pred_y_)
    return error


"""### 2.3.4 Non-Linearity of the Nearest Neighbor Classifier (N4)"""

def ft_N4(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray, 
          metric: str = "euclidean", p=2, n_neighbors=1) -> np.ndarray:
    interp_X = []
    interp_y = []

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    
    for idx in cls_index:
        X_ = X[idx]
        A = np.random.choice(X_.shape[0], X_.shape[0])
        A = X_[A]
        B = np.random.choice(X_.shape[0], X_.shape[0])
        B = X_[B]
        delta = np.random.ranf(X_.shape)
        interp_X_ = A + ((B - A) * delta)
        interp_y_ = y[idx]
        interp_X.append(interp_X_)
        interp_y.append(interp_y_)
    
    # join the datasets
    X_test = np.concatenate(interp_X)
    y_test = np.concatenate(interp_y)
    
    dst = np.asarray(gower.gower_matrix(X, X_test))
    y_test_ = []
    pred_y_ = []
    for test_index in range(X_test.shape[0]):
        fnn_index = np.argmin(dst[test_index])
        y_test_.append(y_test[test_index])
        pred_y_.append(y[fnn_index])
    
    error = 1 - accuracy_score(y_test_, pred_y_)
    return error

"""### 2.3.6 Local Set Average Cardinality (LSC)"""

def LS_i (X: np.ndarray, y: np.ndarray, i: int):
    dst = np.asarray(gower.gower_matrix(X))
    dist_enemy = inter(X, y, dst, i)
    counter = 0
    for index in range(X.shape[0]):
        if dst[i][index] < dist_enemy:
            counter += 1
    return counter

def LSC (X: np.ndarray, y: np.ndarray) -> float:
    n = np.shape(X)[0]
    x = [LS_i(X, y, i) for i in range(n)]
#     print(np.sum(x))
    return 1 - np.sum(x)/n**2