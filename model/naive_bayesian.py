from logger.Logger import logger

class NaiveBayes:

    def __init__(self,df,classified):
        self.df = df
        self.classified = classified
        self.class_probs = self.get_class_probs()
        self.feature_probs = {}
        self.classes = df[classified].unique()
        self.features = [col for col in df.columns if col != classified]
        logger.info("NaiveBayes initialized")

    def get_class_probs(self):
        logger.debug("Calculating class probabilities")
        class_counts = self.df[self.classified].value_counts()
        total = len(self.df)
        return (class_counts / total).to_dict()


    def fit(self):
        logger.info("Fitting NaiveBayes model")
        for feature in self.features:
            self.feature_probs[feature] ={}
            for cls in self.classes:
                subset = self.df[self.df[self.classified] == cls]
                val_counts = subset[feature].value_counts()

                total_cls = len(subset)
                unique_vals = self.df[feature].unique()
                smoothed_probs = {
                    val: (val_counts.get(val, 0) + 1) / (total_cls + len(unique_vals))
                    for val in unique_vals
                }
                self.feature_probs[feature][cls] = smoothed_probs
        logger.info("Model fitting complete")

