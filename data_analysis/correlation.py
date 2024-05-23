import pandas as pd
from matplotlib import pyplot as plt

mushrooms_df = pd.read_csv('./data/mushrooms.csv')

# Make the "class" = 1 if p else 0
# All other columns can be factorized
for column in mushrooms_df.columns:
    if column != 'class':
        mushrooms_df[column] = pd.factorize(mushrooms_df[column])[0]
    else:
        mushrooms_df[column] = mushrooms_df[column].apply(
            lambda x: 1 if x == 'p' else 0)

mushrooms_df.rename(columns={'class': 'poisonous'}, inplace=True)
print(mushrooms_df.head())

corr_matrix = mushrooms_df.corr()["poisonous"]

# output to file
corr_matrix.to_csv('correlation_2.csv')
# print(mushrooms_df.head())

# # Get header names excluding the class column
# features = mushrooms_df.columns[1:]

# print('len features: ', len(features))

# print(features)
# # One hot encoding
# mushrooms_df = pd.get_dummies(mushrooms_df, columns=features, dtype=int)

# # Change "class" to "p" = 1, "e" = 0
# mushrooms_df['class'] = mushrooms_df['class'].apply(
#     lambda x: 1 if x == 'p' else 0)

# # Rename "class"to "poisonous"
# mushrooms_df.rename(columns={'class': 'poisonous'}, inplace=True)

# corr_matrix = mushrooms_df.corr()['poisonous']

# # output to file
# corr_matrix.to_csv('correlation.csv')
