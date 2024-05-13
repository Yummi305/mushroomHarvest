import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def preprocessData(data: pd.DataFrame) -> pd.DataFrame:
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


def main():
    mushrooms_csv = './data/mushrooms.csv'

    # Load the data
    mushrooms_df = pd.read_csv(mushrooms_csv)

    # Preprocess the data
    mushrooms_df = preprocessData(mushrooms_df)

    X = mushrooms_df.drop('poisonous', axis=1)
    y = mushrooms_df['poisonous']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)

    # Write to file
    model_path = './pretrained_models/logreg_model_v2.pkl'

    pickle.dump(logreg, open(model_path, 'wb'))

    print('Model saved to:', model_path)


if __name__ == "__main__":
    main()
