import typing as t
import numpy as np
import gower


"""## 2.4 Network measures"""

#def produce_adjucent_matrix(X, Y):
    #F = gower.gower_matrix(X)
    #G = gower.gower_matrix(X)
    #connected = np.zeros((F.shape))
    #for k in range(len(X)):
        #l = []
        #G[k].sort()
        #for i in range(1,int(0.15*X.shape[0])+1):
            #for j in range(len(X)):
                #if G[k,i] == F[k,j]:
                    #try:
                        #l.index(j)
                    #except:
                        #if Y[k] == Y[j]:
                            #connected[k, j] = 1
                            #l.append(j)
                        #break
    #return connected

def produce_adjucent_matrix(X, Y):
  D = gower.gower_matrix(X, X)
  F = D < 0.15
  for i in range(X.shape[0]):
    for j in range(X.shape[0]):
      if F[i, j] == True and Y[i] != Y[j]:
        F[i, j] = False
  return F


"""### 2.4.1 Average density of the network (Density)"""

def ft_Density(connected, X: np.ndarray, Y: np.ndarray) -> float:
    # connected = produce_adjucent_matrix(X, Y)
    return 1-(connected.sum()/(X.shape[0]*(X.shape[0]-1)))


"""### 2.4.2 Clustering coefficient (ClsCoef)"""

def ft_ClsCoef(connected, X: np.ndarray, Y: np.ndarray) -> float:
    # connected = produce_adjucent_matrix(X, Y)
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


"""### 2.4.3 Hub score (Hubs)"""

def ft_Hubs(connected, X: np.ndarray, Y: np.ndarray, k: int) -> float:
    # connected = produce_adjucent_matrix(X, Y) 
#     connected = test(X, Y)
    y0 = np.ones(connected.shape[0])
    r = np.matmul(connected, y0)
    r = r/np.sqrt(np.power(r, 2).sum())
    for i in range(k):
        r = np.matmul(connected, r)
        r = r/np.sqrt(np.power(r, 2).sum())
    return 1 - r.mean()

