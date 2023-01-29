# Classification-Complexity-Measures

In this repository, the measures reviewed in article [**"How Complex is your classification problem? A survey on measuring classification complexity"**](https://arxiv.org/abs/1808.03591) have been implemented.

## Feature-based Measures: ***overlapping.py***
* Maximum Fisher’s Discriminant Ratio (F1)
* The Directional-vector Maximum Fisher’s Discriminant Ratio (F1v)
* Volume of Overlapping Region (F2)
* Maximum Individual Feature Efficiency (F3)
* Collective Feature Efficiency (F4)


## Measures of Linearity: ***linearity.py***
* Sum of the Error Distance by Linear Programming (L1)
* Error Rate of Linear Classifier (L2)
* Non-Linearity of a Linear Classifier (L3)


## Neighborhood Measures: ***neighborhood.py***
* Fraction of Borderline Points (N1)
* Ratio of Intra/Extra Class Nearest Neighbor Distance (N2)
* Error Rate of the Nearest Neighbor Classifier (N3)
* Non-Linearity of the Nearest Neighbor Classifier (N4)
* Fraction of Hyperspheres Covering Data (T1)
* Local Set Average Cardinality (LSC)


## Network Measures: ***network.py***
* Average density of the network (Density)
* Clustering coefficient (ClsCoef)
* Hub score (Hubs)


## Dimensionality Measures: ***dimensionality.py***
* Average number of features per dimension (T2)
* Average number of PCA dimensions per points (T3)
* Ratio of the PCA Dimension to the Original Dimension (T4)


## Class Imbalance Measures: ***balance.py***
* Entropy of class proportions (C1)
* Imbalance ratio (C2)
