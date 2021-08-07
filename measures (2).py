import typing as t
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import random
import gower
from sklearn.svm import SVC

def precompute_fx(X: np.ndarray, Y: np.ndarray) -> t.Dict[str, t.Any]:
  prepcomp_vals = {}    
  classes, class_freqs = np.unique(Y, return_counts=True)
  cls_index = [np.equal(Y, i) for i in range(classes.shape[0])]
  cls_n_ex = list(class_freqs)
  ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))
  prepcomp_vals["ovo_comb"] = ovo_comb
  prepcomp_vals["cls_index"] = cls_index
  prepcomp_vals["cls_n_ex"] = cls_n_ex
  return prepcomp_vals

def numerator (X: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray, f: int) -> float:
    summation = 0
    for j in range(len(cls_n_ex)):
        for k in range(len(cls_n_ex)): 
            summation += cls_n_ex[j]/sum(cls_n_ex) * cls_n_ex[k]/sum(cls_n_ex) * np.power((np.mean(X[cls_index[j], f])- np.mean(X[cls_index[k], f])), 2)
    return summation
# according to aquation(2)

def denominator (X: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray, f: int) -> float:
    summation = 0
    for j in range(len(cls_n_ex)):
        summation += cls_n_ex[j]/sum(cls_n_ex) * np.power(np.std(X[cls_index[j], f]), 2)
    return summation
# according to aquation(2)

def numerator (X: np.ndarray, cls_index, cls_n_ex, i) -> float:
    return np.sum([cls_n_ex[j]*np.power((np.mean(X[cls_index[j], i])-np.mean(X[:, i], axis=0)),2) for j in range (len(cls_index))])
# according to aquation(3)

def denominator (X: np.ndarray, cls_index, cls_n_ex, i) -> float:
    return np.sum([np.sum(np.power(X[cls_index[j], i]-np.mean(X[cls_index[j], i], axis=0), 2)) for j in range(0, len(cls_n_ex))])
# according to aquation(3)

def compute_rfi (X: np.ndarray, cls_index, cls_n_ex) -> float:
  ls = []
  for i in range(np.shape(X)[1]):
    if denominator(X, cls_index, cls_n_ex, i)!= 0:
      ls.append(numerator (X, cls_index, cls_n_ex, i)/denominator(X, cls_index, cls_n_ex, i))
  return ls

def ft_F1(X: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> float:
    return 1/(1 + np.max(compute_rfi (X, cls_index, cls_n_ex)))

def dVector(X: np.ndarray, y_class1: np.ndarray, y_class2: np.ndarray) -> float:
    X_class1 = X[y_class1]; u_class1 = np.mean(X_class1, axis= 0)
    X_class2 = X[y_class2]; u_class2 = np.mean(X_class2, axis= 0)
    
    W = ((np.shape(X_class1)[0]/ (np.shape(X_class1)[0] + np.shape(X_class2)[0]))* np.cov(X_class1.T))      + (np.shape(X_class2)[0]/(np.shape(X_class1)[0] + (np.shape(X_class2)[0])) * np.cov(X_class2.T))
    try:
      d = np.dot(np.linalg.inv(W), (u_class1 - u_class2))
    except:
      return 0
    
    B = np.dot((u_class1 - u_class2),((u_class1 - u_class2).T))
    
    return np.dot(np.dot(d.T, B), d)/ np.dot(np.dot(d.T, W), d)




def ft_F1v(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray) ->float:
    df_list = []
    
    for idx1, idx2 in ovo_comb:
        y_class1 = cls_index[idx1]
        y_class2 = cls_index[idx2]
        dF = dVector(X, y_class1, y_class2)
        if dF == 0:
          continue
        df_list.append(1/(1+dF))
        
    return np.mean(df_list)




def _minmax(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the minimum of the maximum values per class
    for all features.
    """
    max_cls = np.zeros((2, X.shape[1]))
    try:
      max_cls[0, :] = np.max(X[class1], axis=0)
    except:
      print()
    try:
      max_cls[1, :] = np.max(X[class2], axis=0)
    except:
      print()
    aux = np.min(max_cls, axis=0)
    
    return aux

def _minmin(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the minimum of the minimum values per class
    for all features.
    """
    min_cls = np.zeros((2, X.shape[1]))
    min_cls[0, :] = np.min(X[class1], axis=0)
    min_cls[1, :] = np.min(X[class2], axis=0)
    aux = np.min(min_cls, axis=0)
    
    return aux

def _maxmin(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the maximum of the minimum values per class
    for all features.
    """
    min_cls = np.zeros((2, X.shape[1]))
    try:
      min_cls[0, :] = np.min(X[class1], axis=0)
    except:
      print()
    try:
      min_cls[1, :] = np.min(X[class2], axis=0)
    except:
      print()
    aux = np.max(min_cls, axis=0)
    
    return aux

def _maxmax(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the maximum of the maximum values per class
    for all features. 
    """
    max_cls = np.zeros((2, X.shape[1]))
    max_cls[0, :] = np.max(X[class1], axis=0)
    max_cls[1, :] = np.max(X[class2], axis=0)
    aux = np.max(max_cls, axis=0)
    return aux

def ft_F2(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray) -> float:
    f2_list = []
    
    for idx1, idx2 in ovo_comb:
        y_class1 = cls_index[idx1]
        y_class2 = cls_index[idx2]
        zero_ = np.zeros(np.shape(X)[1])
        overlap_ = np.maximum(zero_, _minmax(X, y_class1, y_class2)-_maxmin(X, y_class1, y_class2))
        range_ = _maxmax(X, y_class1, y_class2)-_minmin(X, y_class1, y_class2)
        ratio = overlap_/range_
        for i in range(ratio.shape[0]):
          if math.isnan(ratio[i]) == True:
            ratio[i] = 1
        f2_list.append(np.prod(ratio))
        
    return np.mean(f2_list)



def _compute_f3(X_: np.ndarray, minmax_: np.ndarray, maxmin_: np.ndarray) -> np.ndarray:
    """ This function computes the F3 complexity measure given minmax and maxmin."""

    overlapped_region_by_feature = np.logical_and(X_ >= maxmin_, X_ <= minmax_)

    n_fi = np.sum(overlapped_region_by_feature, axis=0)
    idx_min = np.argmin(n_fi)

    return idx_min, n_fi, overlapped_region_by_feature

def ft_F3(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> np.ndarray:
    
    f3 = []
    for idx1, idx2 in ovo_comb:
      mima = _minmax(X, cls_index[idx1], cls_index[idx2])
      mami = _maxmin(X, cls_index[idx1], cls_index[idx2])
      ov = np.logical_and(X > mami, X < mima)
      n_fi = np.sum(ov, axis=0)
      idx_min = np.argmin(n_fi)
      f3.append(n_fi[idx_min] / (cls_n_ex[idx1] + cls_n_ex[idx2]))

    return np.mean(f3)
    


def ft_F4(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> np.ndarray:

    f4 = []
    for idx1, idx2 in ovo_comb:
        aux = 0

        y_class1 = cls_index[idx1]
        y_class2 = cls_index[idx2]
        sub_set = np.logical_or(y_class1, y_class2)
        y_class1 = y_class1[sub_set]
        y_class2 = y_class2[sub_set]
        X_ = X[sub_set, :]
        # X_ = X[np.logical_or(y_class1, y_class2),:]
    
        while X_.shape[1] > 0 and X_.shape[0] > 0:
            # True if the example is in the overlapping region
            idx_min, _, overlapped_region_by_feature = _compute_f3(X_,_minmax(X_, y_class1, y_class2),_maxmin(X_, y_class1, y_class2))
            # boolean that if True, this example is in the overlapping region
            overlapped_region = overlapped_region_by_feature[:, idx_min]
            # removing the non overlapped features
            X_ = X_[overlapped_region, :]
            y_class1 = y_class1[overlapped_region]
            y_class2 = y_class2[overlapped_region]
            if X_.shape[0] > 0:
                aux = X_.shape[0]
            else:
                aux = 0
            # removing the most efficient feature
            X_ = np.delete(X_, idx_min, axis=1)
        f4.append(aux/(cls_n_ex[idx1] + cls_n_ex[idx2]))
    return np.mean(f4)



def ft_R1(model, X: np.ndarray, Y: np.ndarray) -> float:
    y = model.decision_function(X)
    w_norm = np.linalg.norm(model.coef_)
    dist = y / w_norm
    Y_predicted = model.predict(X)
    distance = 0
    for i in range(Y.shape[0]):
        if(Y_predicted[i] != Y[i]):
            distance += dist[i, 0] + dist[i, 1] + dist[i, 2]
    SumErrorDist = distance/(Y.shape[0]*3)
    L1 = SumErrorDist/(SumErrorDist + 1)
    return L1


def ft_R2(model, X: np.ndarray, Y: np.ndarray) -> float:
    Y_predicted = model.predict(X)
    counter = 0
    for i in range(Y.shape[0]):
        if(Y_predicted[i] != Y[i]):
            counter += 1
    L2 = counter / Y.shape[0]
    return L2




def ft_R3(model, X: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> float:
    temp_x = []
    temp_y = []
    for i in range(len(cls_n_ex)):
        for j in range(cls_n_ex[i]):
            index = np.random.choice(cls_n_ex[i], 2, replace=False)
            points = X[cls_index[i]][index]
            rand = random.uniform(0, 1)
            temp_x.append(((points[1] - points[0])*rand) + points[0])
            temp_y.append(i)

    temp_x = np.asarray(temp_x)
    temp_y = np.asarray(temp_y)
    R3 = ft_R2(model, temp_x, temp_y) 
    return R3



def ft_N1(X: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    
    # 0-1 scaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_ = scaler.transform(X)

    # compute the distance matrix and the minimum spanning tree.
    dist_m = np.triu(distance.cdist(X_, X_, metric), k=1)
    mst = minimum_spanning_tree(dist_m)
    node_i, node_j = np.where(mst.toarray() > 0)
    # which edges have nodes with different class
    which_have_diff_cls = y[node_i] != y[node_j]
    # number of different vertices connected
    aux = np.unique(np.concatenate([node_i[which_have_diff_cls],node_j[which_have_diff_cls]])).shape[0]
    return aux/X.shape[0]



def extra_nearest (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray, 
                   i: int, metric: str = "euclidean", n_neighbors=1) :
    " This function computes the distance from a point x_i to their nearest enemy"
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    
    X_ = X[np.logical_not(cls_index[y[i]])]
    y_ = y[np.logical_not(cls_index[y[i]])]
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    neigh.fit(X_, y_) 
    dist_enemy, pos_enemy = neigh.kneighbors([X[i, :]])
    dist_enemy = np.reshape(dist_enemy, (n_neighbors,))
    pos_enemy_ = np.reshape(pos_enemy, (n_neighbors,))
    query = X_[pos_enemy_, :]
    pos_enemy = np.where(np.all(X==query,axis=1))
    pos_enemy = np.reshape(pos_enemy, (n_neighbors,))
    return dist_enemy, pos_enemy

def intra_nearest (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray,
                                  i: int, metric: str = "euclidean", n_neighbors=1) :
    " This function computes the distance from a point x_i to their nearest neighboor from its own class"
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    
    query = X[i, :]
    label_query = y[i]
    X_ = X[cls_index[label_query]]
    y_ = y[cls_index[label_query]]
    
    pos_query = np.where(np.all(X_==query,axis=1))
    X_ = np.delete(X_, pos_query, axis = 0)
    y_ = np.delete(y_, pos_query, axis = 0) 
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    neigh.fit(X_, y_) 
    dist_neigh, pos_neigh = neigh.kneighbors([X[i, :]])
    dist_neigh = np.reshape(dist_neigh, (n_neighbors,))
    pos_neigh = np.reshape(pos_neigh, (n_neighbors,))
    return dist_neigh, pos_neigh

def intra_extra(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray) -> float:
    intra = np.sum([intra_nearest (X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    extra = np.sum([extra_nearest (X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    return intra/extra

def ft_N2 (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray) -> float:
    intra_extra_ = intra_extra(X, y, cls_index)
    return intra_extra_/(1+intra_extra_)




def ft_N3 (X: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> float:
    
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_ = scaler.transform(X)
    loo = LeaveOneOut()
    loo.get_n_splits(X_, y)
    
    y_test_ = []
    pred_y_ = []
    for train_index, test_index in loo.split(X_):
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = KNeighborsClassifier(n_neighbors=1, metric=metric)
        model.fit(X_train, y_train)
        pred_y = model.predict(X_test)
        y_test_.append(y_test)
        pred_y_.append(pred_y)
    
    error = 1 - accuracy_score(y_test_, pred_y_)
    return error




def ft_N4(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray, 
          metric: str = "euclidean", p=2, n_neighbors=1) -> np.ndarray:
    interp_X = []
    interp_y = []

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    
    for idx in cls_index:
        #creates a new dataset by interpolating pairs of training examples of the same class.
        X_ = X[idx]
        #two examples from the same class are chosen randomly and
        #they are linearly interpolated (with random coefficients), producing a new example.
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
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric=metric).fit(X, y)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    return error




def distance_matrix (X: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    dist = distance.cdist(X, X, 'euclidean')
    return dist


def radios (D: np.ndarray, y: np.ndarray, X: np.ndarray, 
                        cls_index:np.ndarray, i: int) -> float:
    d_i, x_j = extra_nearest(X, y, cls_index, i)
    d_j, x_k = extra_nearest(X, y, cls_index, x_j[0])
    if (i == x_k[0]):
        return d_i/2
    else :
        d_t = radios (D, y, X, cls_index, x_j[0]) 
        var = d_i - d_t
        return d_i - d_t

def hyperspher (D: np.ndarray, y: np.ndarray, X: np.ndarray, cls_index:np.ndarray) -> np.ndarray:
    aux = [radios(D, y, X, cls_index, i) for i in range(X.shape[0])]
    return aux

def ft_T1(X: np.ndarray, Y: np.ndarray, cls_index: np.ndarray, eps: float) -> float:
    D = distance_matrix(X)
    hyper_list = hyperspher(D, Y, X, cls_index)
    
    for i in range(len(hyper_list)):
        flag = True
        while flag == True:
            r = hyper_list[i] + eps
            d, p = extra_nearest(X, Y, cls_index, i)
            if d > hyper_list[i] + 2*eps:
                hyper_list[i] = hyper_list[i] + eps
            else:
                flag = False
    
    checking = np.zeros(len(hyper_list))
    for i in range(len(hyper_list)):
        for j in range(len(hyper_list)):
            if Y[i] == Y[j] and i != j:
#                 print(i,j,D[i, j], hyper_list[i], hyper_list[j])
                if D[i, j] + hyper_list[i] <= hyper_list[j]:
                    checking[i] = 1
                    break

    counter = 0
    backup = hyper_list
    remnants = np.zeros(len(hyper_list))
    while counter < 150:
        if max(backup) == 0:
            break
        index = hyper_list.index(max(backup))
        for i in range(len(hyper_list)):
            if D[index, i] <= backup[index] and checking[i] != 1:
                counter += 1
                checking[i] = 1
                remnants[index] = 1
            backup[index] = 0
    num = remnants.sum()
    return num/len(X)

def LS_i (X: np.ndarray, y: np.ndarray, i: int, cls_index, metric: str = "euclidean"):
    dist_enemy, pos_enemy = extra_nearest(X, y, cls_index, i)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    dist_ = distance.cdist(X, [X[i, :]], metric=metric)
    X_j = dist_[np.logical_and(dist_ < dist_enemy, dist_ != 0)]
    return X_j

def LSC (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray) -> float:
    n = np.shape(X)[0]
    x = [np.shape(LS_i(X, y, i, cls_index)) for i in range(n)]
    return 1 - np.sum(x)/n**2

def produce_adjucent_matrix(X, Y: np.ndarray) -> np.ndarray:
    F = gower.gower_matrix(X)
    connected = np.zeros((F.shape))
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if F[i, j] <= 0.15 and i != j:
                connected[i, j] = 1

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if Y[i] != Y[j]:
                connected[i, j] = 0
    return connected

def test(X, Y):
    F = gower.gower_matrix(X)
    G = gower.gower_matrix(X)
    connected = np.zeros((F.shape))
    for k in range(len(X)):
        l = []
        G[k].sort()
        for i in range(1,int(0.15*X.shape[0])+1):
            for j in range(len(X)):
                if G[k,i] == F[k,j]:
                    try:
                        l.index(j)
                    except:
                        if Y[k] == Y[j]:
    #                         print(k,j,F[k,j],Y[k], Y[j])
                            connected[k, j] = 1
                            l.append(j)
                        break
    return connected



def ft_Density(X: np.ndarray, Y: np.ndarray) -> float:
    F = gower.gower_matrix(X)
    connected = np.zeros((F.shape))
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if F[i, j] <= 0.15 and i != j:
                connected[i, j] = 1

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if Y[i] != Y[j]:
                connected[i, j] = 0
                
    E = connected.sum()/2
    aux = 1 - (2*E)/(F.shape[0]*(F.shape[0]-1))
    return aux



def ft_ClsCoef(X: np.ndarray, Y: np.ndarray) -> float:
    connected = produce_adjucent_matrix(X, Y)
#     connected = test(X, Y)
    summation = 0
    for i in range(connected.shape[0]):
        neighbors = []
        for k in range(connected[i, :].shape[0]):
            if connected[i, k] == 1:
                neighbors.append(k)
        edge = 0
        for m in range(len(neighbors)):
            for n in range(m + 1, len(neighbors)):
                if connected[m, n] == 1:
                    edge = edge + 1
        if len(neighbors) > 1:
            summation = summation + (2*edge)/(len(neighbors)*(len(neighbors) - 1))
    aux = 1 - summation/connected.shape[0]
    return aux



def ft_Hubs2(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    connected = produce_adjucent_matrix(X, Y)
    hubs = []
    for i in range(connected.shape[0]):
        hubs.append(sum(connected[i, :]))
    hubs = hubs/sum(hubs)
    for i in range(k):
        for m in range(connected.shape[0]):
            summation = 0
            for n in range(connected.shape[1]):
                if(connected[m, n] == 1):
                    summation = summation + hubs[n]
            hubs[m] = summation
        hubs = hubs/sum(hubs)
    return 1-sum(hubs)/len(hubs)



def ft_Hubs(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    connected = produce_adjucent_matrix(X, Y) 
#     connected = test(X, Y)
    y0 = np.ones(connected.shape[0])
    r = np.matmul(connected, y0)
    r = r/np.sqrt(np.power(r, 2).sum())
    for i in range(k):
        r = np.matmul(connected, r)
        r = r/np.sqrt(np.power(r, 2).sum())
    return 1 - r.mean()


def precompute_pca_tx(X: np.ndarray) -> t.Dict[str, t.Any]:
    prepcomp_vals = {}

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    pca = PCA(n_components=0.95)
    pca.fit(X)

    m_ = pca.explained_variance_ratio_.shape[0]
    m = X.shape[1]
    n = X.shape[0]

    prepcomp_vals["m_"] = m_
    prepcomp_vals["m"] = m
    prepcomp_vals["n"] = n

    return prepcomp_vals


def ft_T2(m: int, n: int) -> float:
    return m/n


def ft_T3(m_: int, n: int) -> float:
    return m_/n



def ft_T4(m: int, m_: int) -> float:
    return m_/m



def ft_C1(cls_n_ex: np.ndarray) -> float:
    nc = len(cls_n_ex)
    n = sum(cls_n_ex)
    summation = 0
    for i in range(nc):
        pi = cls_n_ex[i]/n
        summation = summation + pi * math.log(pi)
    aux = 1 + summation / math.log(nc)
    return aux



def ft_C2(cls_n_ex: np.ndarray) -> float:
    nc = len(cls_n_ex)
    n = sum(cls_n_ex)
    summation = 0
    for i in range(nc):
        summation = summation + cls_n_ex[i]/(n - cls_n_ex[i])
    aux = ((nc - 1)/nc) * summation
    aux = 1 - (1/aux)
    return aux


def compute_all_measures(X, Y):
  precompute = precompute_fx(X, Y)
  cls_index = precompute['cls_index']
  cls_n_ex = precompute['cls_n_ex']
  ovo_comb = precompute['ovo_comb']
  precomp_pca = precompute_pca_tx(X)
  m = precomp_pca['m']
  n = precomp_pca['n']
  m_ = precomp_pca['m_']

  model = SVC(kernel = 'linear')
  model.fit(X, Y);

  print("F1 score: ",ft_F1(X, cls_index,cls_n_ex))
  print("F1v score: ",ft_F1v(X, ovo_comb, cls_index))
  print("F2 score: ",ft_F2(X, ovo_comb, cls_index))
  print("F3 score: ",ft_F3(X, ovo_comb, cls_index, cls_n_ex))
  print("F4 score: ",ft_F4(X, ovo_comb, cls_index, cls_n_ex))
  print("R1 score: ",ft_R1(model, X, Y))
  print("R2 score: ",ft_R2(model, X, Y))
  print("R3 score: ",ft_R3(model, X, cls_index, cls_n_ex))
  print("N1 score: ",ft_N1(X, Y))
  print("N2 score: ",ft_N2(X, Y, cls_index))
  print("N3 score: ",ft_N3(X, Y))
  print("N4 score: ",ft_N4(X, Y, cls_index))
  # print("T1 score: ",ft_T1(X, Y, cls_index, 0.001))
  print("LSC score: ",LSC(X, Y, cls_index))
  print("Density score: ",ft_Density(X, Y))
  print("ClsCoef score: ",ft_ClsCoef(X, Y))
  print("Hubs score: ",ft_Hubs(X, Y, 6))
  print("T2 score: ",ft_T2(m, n))
  print("T3 score: ",ft_T3(m, m_))
  print("T4 score: ",ft_T4(m, n))
  print("C1 score: ",ft_C1(cls_n_ex))
  print("C2 score: ",ft_C2(cls_n_ex))


