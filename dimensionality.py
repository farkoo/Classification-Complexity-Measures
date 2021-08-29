import typing as t
import numpy as np
from sklearn.decomposition import PCA

"""## 2.5 Dimensionality Measures"""

"""### 2.5.1 Average number of features per points (T2)"""

def ft_T2(m: int, n: int) -> float:
    return m/n

"""### 2.5.2 Average number of PCA dimensions per points (T3)"""

def ft_T3(m_: int, n: int) -> float:
    return m_/n

"""### 2.5.3 Ratio of the PCA Dimension to the Original Dimension (T4)"""

def ft_T4(m: int, m_: int) -> float:
    return m_/m
