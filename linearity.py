import typing as t
import numpy as np
from sklearn.svm import SVC
import random

"""## 2.2 Measures of Linearity

### 2.2.1 Sum of the Error Distance by Linear Programming (L1)
"""

def ft_R1(model, X: np.ndarray, Y: np.ndarray) -> float:
    y = model.decision_function(X)
    w_norm = np.linalg.norm(model.coef_)
    dist = y / w_norm
    Y_predicted = model.predict(X)
    distance = 0
    for i in range(Y.shape[0]):
        
        if(Y_predicted[i] != Y[i]):
           distance += abs(dist[i, Y[i]])
          #  print(i,Y_predicted[i], Y[i], dist[i,:])
    SumErrorDist = distance/(Y.shape[0])
#             distance += dist[i, 0] + dist[i, 1] + dist[i, 2]
#     SumErrorDist = distance/(Y.shape[0]*3)
    
    L1 = SumErrorDist/(SumErrorDist + 1)
    
    return L1


"""### 2.2.2 Error Rate of Linear Classifier (L2)"""

def ft_R2(model, X: np.ndarray, Y: np.ndarray) -> float:
    Y_predicted = model.predict(X)
    counter = 0
    for i in range(Y.shape[0]):
        if(Y_predicted[i] != Y[i]):
            counter += 1
    L2 = counter / Y.shape[0]
    return L2

"""### 2.2.3 Non-Linearity of a Linear Classifier (L3)"""

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
