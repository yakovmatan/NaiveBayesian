from data.loadCsv import CSVLoader
from logger.Logger import logger
from model.naive_bayesian import NaiveBayes


class Prediction:

    def __init__(self,model: NaiveBayes):
        self.model = model

    def prediction(self, row):
        logger.debug(f"Predicting row: {row}")
        probs = {}
        for cls in self.model.classes:
            prob = self.model.class_probs[cls]
            for feature in self.model.features:
                val = row[feature]
                feature_dict = self.model.feature_probs[feature][cls]
                prob *= feature_dict.get(val)
            probs[cls] = prob
        prediction = max(probs, key=probs.get)
        logger.info(f"Prediction: {prediction}")
        return prediction




