import typing as t
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def precompute_fx(X: np.ndarray, Y: np.ndarray) -> t.Dict[str, t.Any]:
  prepcomp_vals = {}    
  classes, class_freqs = np.unique(Y, return_counts=True)
  cls_index = [np.equal(Y, i) for i in range(classes.shape[0])]
  cls_n_ex = list(class_freqs)
  ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))
  cls_index = np.asarray(cls_index).reshape(len(cls_n_ex),X.shape[0])
  prepcomp_vals["ovo_comb"] = ovo_comb
  prepcomp_vals["cls_index"] = cls_index
  prepcomp_vals["cls_n_ex"] = cls_n_ex
  return prepcomp_vals

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

from overlapping import *
from linearity import *
from neighborhood import *
from network import * 
from dimensionality import *
from balance import *

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

  connected = produce_adjucent_matrix(X, Y)

  print("F1 score: ", ft_F1(X, cls_index,cls_n_ex))
  print("F1v score: ",ft_F1v(X, ovo_comb, cls_index))
  print("F2 score: ",ft_F2(X, ovo_comb, cls_index))
  print("F3 score: ",ft_F3(X, ovo_comb, cls_index, cls_n_ex))
  print("F4 score: ",ft_F4(X, ovo_comb, cls_index, cls_n_ex))
  print("R1 score: ",ft_R1(model, X, Y))
  print("R2 score: ",ft_R2(model, X, Y))
  print("R3 score: ",ft_R3(model, X, cls_index, cls_n_ex))
  print("N1 score: ",ft_N1(X, Y))
  print("N2 score: ",ft_N2(X, Y))
  print("N3 score: ",ft_N3(X, Y))
  print("N4 score: ",ft_N4(X, Y, cls_index))
  print("T1 score: ",ft_T1(X, Y))
  print("LSC score: ",LSC(X, Y))
  print("Density score: ",ft_Density(connected, X, Y))
  print("ClsCoef score: ",ft_ClsCoef(connected, X, Y))
  print("Hubs score: ",ft_Hubs(connected, X, Y, 6))
  print("T2 score: ",ft_T2(m, n))
  print("T3 score: ",ft_T3(m_, n))
  print("T4 score: ",ft_T4(m, m_))
  print("C1 score: ",ft_C1(cls_n_ex))
  print("C2 score: ",ft_C2(cls_n_ex))


