import pandas as pd
import prince
from matplotlib import pyplot as plt

mushrooms_df = pd.read_csv('./data/mushrooms.csv')

# Get header names excluding the class column
features = mushrooms_df.columns[1:]
features_df = mushrooms_df.iloc[:, 1:]

mca = prince.MCA(
    n_components=100,
    n_iter=10,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)

mca = mca.fit(features_df)

cummulative_var = mca.cumulative_percentage_of_variance_

cummulative_var = cummulative_var / 100

print('First index where the cumulative sum is greater than 0.95:',
      next(i for i, x in enumerate(cummulative_var) if x > 0.95))

print('First index where the cumulative sum is greater than 0.99:',
      next(i for i, x in enumerate(cummulative_var) if x > 0.99))

print(cummulative_var)

fig = plt.figure()

plt.plot(cummulative_var)
# Add a point at x=59, y=0.95, with lines to the axes
plt.scatter(59, 0.95, color='r')

# Add a point at x=71, y=0.99
plt.scatter(71, 0.99, color='g')

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

plt.legend(['Cumulative explained variance', '0.95', '0.99'])
plt.title('MCA Cumulative Variance')

# Save image
plt.savefig('mca_cumulative_variance.png')
