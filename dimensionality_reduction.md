# PCA or MCA

Initially we explored the possibility of reducing the dimensionality of the dataset to keep only the most significant components.

In order to reduce the dimensionality of categorical data we had to first one hot encode the existing features. This resulted in an initial increase of features columns from 22 to 117.

Next PCA was performed on the dataset. In order to maintain 95% of the variance we were able to reduce the dimensionality from 117 to 40. To maintain 99% of the variance 57 dimensions were required.
