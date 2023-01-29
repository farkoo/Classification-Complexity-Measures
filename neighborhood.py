import typing as t
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import MinMaxScaler
import gower
import itertools
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import scipy.spatial

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

def hyperspheres_radius(nearest_enemy_ind, nearest_enemy_dist):
    def recurse_radius(ind_inst):
        if radius[ind_inst] >= 0.0:
            return radius[ind_inst]
        ind_enemy = nearest_enemy_ind[ind_inst]
        ind_enemy = int(ind_enemy)
        if (ind_inst == nearest_enemy_ind[ind_enemy]):
            radius[ind_enemy] = radius[ind_inst] = (0.5 * nearest_enemy_dist[ind_inst])
            return radius[ind_inst]
        radius[ind_inst] = 0.0
        radius_enemy = recurse_radius(ind_inst=ind_enemy)
        radius[ind_inst] = abs(nearest_enemy_dist[ind_inst] - radius_enemy)
        return radius[ind_inst]
    radius = np.full(nearest_enemy_ind.size, fill_value=-1.0, dtype=float)
    for ind in np.arange(radius.size):
        if radius[ind] < 0.0:
            recurse_radius(ind_inst=ind)
    return radius

def hyper_in(center_a, center_b, radius_a, radius_b,):
    upper_a, lower_a = center_a + radius_a, center_a - radius_a
    upper_b, lower_b = center_b + radius_b, center_b - radius_b
    for ind in np.arange(center_a.size):
        if (upper_a[ind] > upper_b[ind]) or (lower_a[ind] < lower_b[ind]):
            return False
    return True

def hyper_final(centers, radius):
    sorted_sphere_inds = np.argsort(radius)
    sphere_inst_num = np.ones(radius.size, dtype=int)
    for ind_a, ind_sphere_a in enumerate(sorted_sphere_inds[:-1]):
        for ind_sphere_b in sorted_sphere_inds[:ind_a:-1]:
            if hyper_in(center_a=centers[ind_sphere_a, :], center_b=centers[ind_sphere_b, :],
                radius_a=radius[ind_sphere_a],radius_b=radius[ind_sphere_b],):
                sphere_inst_num[ind_sphere_b] += sphere_inst_num[ind_sphere_a]
                sphere_inst_num[ind_sphere_a] = 0
                break
    return sphere_inst_num

"""### 2.3.5 Fraction of Hyperspheres Covering Data (T1)"""

def ft_T1(X, Y):
    nearest_enemy_ind = np.zeros(X.shape[0])
    nearest_enemy_dist = np.zeros(X.shape[0])
    dst = scipy.spatial.distance.cdist(X, X, metric="minkowski", p=2)
    # dst = gower.gower_matrix(X)
    for i in range(X.shape[0]):
        minn = inter(X, Y, dst, i)
        indexx = np.where(dst[i]==minn)[0]
        nearest_enemy_dist[i] = minn
        nearest_enemy_ind[i] = indexx[0]
    radius = hyperspheres_radius(nearest_enemy_ind=nearest_enemy_ind,nearest_enemy_dist=nearest_enemy_dist,)
    sphere_inst_count = hyper_final(centers=X, radius=radius)
    t1 = sphere_inst_count[sphere_inst_count > 0].shape[0]/(X.shape[0]*np.unique(Y).shape[0])
    return t1
    
"""### 2.3.6 Local Set Average Cardinality (LSC)"""

def LSC (X: np.ndarray, Y: np.ndarray) -> float:
  nearest_enemy_dist = np.zeros(X.shape[0])
  dst = gower.gower_matrix(X)
  for i in range(X.shape[0]):
    minn = inter(X, Y, dst, i)
    nearest_enemy_dist[i] = minn
    lsc = 1.0 - np.sum(dst < nearest_enemy_dist) / (X.shape[0] ** 2)
  return lsc
