import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

mushrooms_df = pd.read_csv('./data/mushrooms.csv')

print(mushrooms_df.head())

# Get header names excluding the class column
features = mushrooms_df.columns[1:]

print('len features: ', len(features))

print(features)
# One hot encoding
mushrooms_df = pd.get_dummies(mushrooms_df, columns=features, dtype=int)
features_df = mushrooms_df.iloc[:, 1:]
print('len features_df after one hot: ', len(features_df.columns))
print(features_df.head())

pca = PCA(n_components=100)

features_pca = pca.fit_transform(features_df)

cummulatives = pca.explained_variance_ratio_.cumsum()

# First index where the cumulative sum is greater than 0.95 and then first index where the cumulative sum is greater than 0.99

print('First index where the cumulative sum is greater than 0.95:',
      next(i for i, x in enumerate(cummulatives) if x > 0.95))

print('First index where the cumulative sum is greater than 0.99:',
      next(i for i, x in enumerate(cummulatives) if x > 0.99))

fig = plt.figure()

plt.plot(cummulatives)
# Add a point at x=39, y=0.95, with lines to the axes
plt.scatter(39, 0.95, color='r')

# Add a point at x=56, y=0.99
plt.scatter(56, 0.99, color='g')

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

plt.legend(['Cumulative explained variance', '0.95', '0.99'])
plt.title('PCA Cumulative Variance')

# Save image
plt.savefig('pca_cumulative_variance.png')
# plt.show()
