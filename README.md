# Customers-Type-Clustering
This project will analyze customers' annual spending amounts (in dollar) of different types of product categories. The goal of this project is to find the best variation (cluster) in the different types of customers.

## Getting Started
This project requires Python 2 and needs the following packages: 
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

## Data Exploration
There are six product categories: 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', and 'Delicatessen'. To get a better understanding of the dataset visually, I constructed a scatter matrix of each of the six product features. I immediately noticed that most data points are concentrated at 0, and the density decreases as the spending amount increases. In addition, all features appear to be right skewed, meaning that majority of the data is on the lower values, and the mean is less than the median. This kind of skewness appears on Fresh vs Fresh, Milk vs Milk, etc. There are also many outliers, and those large values make it less obvious for us to see what exactly the distribution of the data are. If we somehow fixed those outliers, the distribution of the graphs might be look more like a normal distribution, at least not as extreme.
```
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
```

## Data Preprocessing
As observed, most categories are skewed to the right and contain outliers. I need to perform data transformation in order to neutralize the skewness in order to maximize the clustering algorithms performance.
To handle with right-skewed data, I apply natural logarithm to every data point:
```
log_data = np.log(data)
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```

To select outliers, I use tukey's method: a data point will be considered as an outlier when it is 1.5 times the interquartile range(IQR). That is, I will consider the data point is an outlier if the value is: 
```
below (Q1) – (1.5 × IQR) or above (Q3) + (1.5 × IQR), where Q1 is 25% percentile, Q3 is 75% percentile, and IQR is Q3 - Q1
```

## Feature Transformation


