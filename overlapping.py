import typing as t
import numpy as np
import math

"""## 2.1 Feature-based Measures

### 2.1.1 Maximum Fisher’s Discriminant Ratio (F1)
"""

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

def compute_rfi (X: np.ndarray, cls_index, cls_n_ex) -> float:
  ls = []
  for i in range(np.shape(X)[1]):
    if denominator(X, cls_index, cls_n_ex, i)!= 0:
      ls.append(numerator (X, cls_index, cls_n_ex, i)/denominator(X, cls_index, cls_n_ex, i))
  return ls

def ft_F1(X: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> float:
    return 1/(1 + np.max(compute_rfi (X, cls_index, cls_n_ex)))

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

"""### 2.1.2 The Directional-vector MaximumFisher's Discreminant Ratio (F1v)"""

def dVector(X: np.ndarray, y_class1: np.ndarray, y_class2: np.ndarray) -> float:
    X_class1 = X[y_class1]; u_class1 = np.mean(X_class1, axis= 0)
    X_class2 = X[y_class2]; u_class2 = np.mean(X_class2, axis= 0)
    W = ((np.shape(X_class1)[0]/ (np.shape(X_class1)[0] + np.shape(X_class2)[0]))* np.cov(X_class1.T)) \
     + (np.shape(X_class2)[0]/(np.shape(X_class1)[0] + (np.shape(X_class2)[0])) * np.cov(X_class2.T))
    try:
      d = np.dot(np.linalg.inv(W), (u_class1 - u_class2))
    except:
      return 0
    B = np.dot((u_class1 - u_class2)[np.newaxis].T,(u_class1 - u_class2).reshape(1,(u_class1 - u_class2).shape[0]))
    dv = np.dot(np.dot(d.T, B), d)/ np.dot(np.dot(d.T, W), d)
    return dv

def ft_F1v(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray) ->float:
    df_list = []
    
    for idx1, idx2 in ovo_comb:
        y_class1 = cls_index[idx1]
        y_class2 = cls_index[idx2]
        dF = dVector(X, y_class1, y_class2)
        if dF == 0:
          continue
        df_list.append(dF)
    aux = np.mean(df_list)
    f1v = 1/(1+aux)
    if math.isnan(f1v) == True:
      f1v = 0.0
    return f1v

"""### 2.1.3 Volume of Overlapping Region (F2)¶"""

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
        
    f2 = np.mean(f2_list)
    if f2 == float("NaN"):
      f2 = 0
    return np.mean(f2_list)

"""### 2.1.4 Maximum Individual Feature Efficiency (F3)"""

def _compute_f3(X_: np.ndarray, minmax_: np.ndarray, maxmin_: np.ndarray) -> np.ndarray:
    """ This function computes the F3 complexity measure given minmax and maxmin."""

    overlapped_region_by_feature = np.logical_and(X_ >= maxmin_, X_ <= minmax_)

    n_fi = np.sum(overlapped_region_by_feature, axis=0)
    idx_min = np.argmin(n_fi)

    return idx_min, n_fi, overlapped_region_by_feature

def ft_F3(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> np.ndarray:
    
    f3 = []
    for idx1, idx2 in ovo_comb:
        idx_min, n_fi, _ = _compute_f3(X, _minmax(X, cls_index[idx1], cls_index[idx2]),
        _maxmin(X, cls_index[idx1], cls_index[idx2]))
    f3.append(n_fi[idx_min] / (cls_n_ex[idx1] + cls_n_ex[idx2]))

    return np.mean(f3)/len(cls_index)


"""### 2.1.5 Colective Feature Efficiency (F4)"""

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
