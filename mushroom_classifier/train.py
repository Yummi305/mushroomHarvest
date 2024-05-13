import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def preprocess_training_data(data: pd.DataFrame) -> pd.DataFrame:
    # Split classes and features
    classes = data['class']
    features = data.drop('class', axis=1)

    # One hot encode features
    features = pd.get_dummies(features)

    # Convert classes to poisonous true/false
    classes = classes == 'p'

    # Rename classes to poisonous
    classes = classes.rename('poisonous')

    # Join classes and features
    preprocessed = pd.concat([classes, features], axis=1)

    return preprocessed


def train_linear_regression(save_to_file, mushrooms_csv):
    model_name = 'logreg_model_v2'

    # Load the data
    mushrooms_df = pd.read_csv(mushrooms_csv)

    # Preprocess the data
    mushrooms_df = preprocess_training_data(mushrooms_df)

    X = mushrooms_df.drop('poisonous', axis=1)
    y = mushrooms_df['poisonous']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)

    # Write to file
    if not save_to_file:
        return logreg, X_train, X_test, y_train, y_test

    model_path = './pretrained_models/' + model_name + '.pkl'

    pickle.dump(logreg, open(model_path, 'wb'))

    print('Model saved to:', model_path)

    return logreg, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_linear_regression(True, './data/mushrooms.csv')
