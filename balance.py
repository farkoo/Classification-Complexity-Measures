import typing as t
import numpy as np
import math

"""## 2.6 Class Imbalance Measures

### 2.6.1 Entropy of class proportions (C1)
"""

def ft_C1(cls_n_ex: np.ndarray) -> float:
    nc = len(cls_n_ex)
    n = sum(cls_n_ex)
    summation = 0
    for i in range(nc):
        pi = cls_n_ex[i]/n
        summation = summation + pi * math.log(pi)
    aux = 1 + summation / math.log(nc)
    return aux


"""### 2.6.2 Imbalance ratio (C2)"""

def ft_C2(cls_n_ex: np.ndarray) -> float:
    nc = len(cls_n_ex)
    n = sum(cls_n_ex)
    summation = 0
    for i in range(nc):
        summation = summation + cls_n_ex[i]/(n - cls_n_ex[i])
    aux = ((nc - 1)/nc) * summation
    aux = 1 - (1/aux)
    return aux

