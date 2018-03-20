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

# Feature scaling
# apply transformation to the data (using natural logarithm to every columns)
log_data = np.log(data)
# apply natural logarithm to the samples
log_samples = np.log(samples)
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    # Display the identified outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    # Note: "~" means not in array: e.g. ~np.array([True]) means False
    print log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]

# indices to be removed
outliers = [66, 95, 96, 218, 338, 357, 86, 98, 154, 356, 75, 325, 161, 183, 109, 128, 142, 187, 233]
# Removing the outliers
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

# Feature Transformation
# Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)


# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# show explained variances
print "Explained variances are shown from higheest to lowest (in percentage): {}, ".format(pca.explained_variance_ratio_*100)

#  Apply PCA by fitting the good data with only two dimensions
# Note: the explained variance will not change even when we change n to 2,
# the first dimension will still have 47.7% explained and second dimension still have 25.07% explained variance
pca = PCA(n_components=2)
pca.fit(good_data)
# Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])