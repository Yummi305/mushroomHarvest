import pickle

import pandas as pd


class MushroomClassifier:
    def __init__(self, model_type='linearregression', model_path='./pretrained_models/logreg_model_v2.pkl'):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None

        if self.model_type == 'linearregression':
            self.model = self.load_linear_regression_model()

    def load_linear_regression_model(self):
        if self.model_type != 'linearregression':
            raise ValueError('Model type not supported')

        return pickle.load(open(self.model_path, 'rb'))

    def preprocess_data(self, data: dict) -> pd.DataFrame:
        data_df = pd.DataFrame(data)

        # Create empty dataframe with columns from model.feature_names_in_
        df_empty = pd.DataFrame(columns=self.model.feature_names_in_)

        # Convert mushroom dictionary to DataFrame
        df_mushroom = pd.get_dummies(data_df)

        # Concat mushroom data to empty DataFrame
        df_appended = pd.concat([df_empty, df_mushroom], axis=0)

        df_appended = df_appended.fillna(0)
        return df_appended

    def is_poisonous(self, mushroom_features: dict) -> bool:
        preprocessed_data = self.preprocess_data(mushroom_features)
        return bool(self.model.predict(preprocessed_data))
