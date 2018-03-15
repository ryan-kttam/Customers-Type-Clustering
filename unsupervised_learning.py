import numpy as np
import pandas as pd


# load the data and prevent errors from occurring
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# data summary (Mean, sd, percentile)
data.describe()

#select three samples from the dataset
indices = [100,200,300]
# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
samples

# understanding the data
# Check whether column "Frozen" can be predicted by other columns
# That is: setting "Frozen" to be a response variable, while all other columns to be predictors
# I will use a decision tree regressor to perform such prediction
new_data = data.drop( ['Frozen'], axis=1)
label = data['Frozen']

# split the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, label, train_size=0.75, random_state=1)

# Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)
# Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print score
# This score is R^2, which implies our features fails to fit and predict the "Frozen" category.
# In order word, the feature of "Frozen" is useful in identifying customers' spending habits.

# Visualizing the feature distributions
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')