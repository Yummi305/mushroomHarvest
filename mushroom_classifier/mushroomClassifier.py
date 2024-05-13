import pickle

from mushroom_classifier.mushroom import Mushroom


class MushroomClassifier:
    def __init__(self, model_type='linearregression', model_path='./pretrained_models/logreg_model_v2.pkl'):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None

        if self.model_type == 'linearregression':
            self.model = self.loadLinearRegressionModel()

    def loadLinearRegressionModel(self):
        if self.model_type != 'linearregression':
            raise ValueError('Model type not supported')

        return pickle.load(open(self.model_path, 'rb'))

    def isPoisonous(self, mushroom: Mushroom) -> bool:
        return self.model.predict(mushroom.features)
