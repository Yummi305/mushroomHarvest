def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
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
