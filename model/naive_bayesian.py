import pandas as pd
from pyexpat import features

from pandas import value_counts


class NaiveBayes:

    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = []
        self.features = []

    def fit(self, df):
        target = df.columns[-1]
        self.classes = df[target].unique()
        self.features = df.columns[:-1]

        class_counts = df[target].value_counts()
        total = len(df)
        self.class_probs = (class_counts / total).to_dict()

        self.feature_probs = {feature: {} for feature in self.features}
        for feature in self.features:
            for cls in self.classes:
                subset = df[df[target] == cls]
                val_counts = subset[feature].value_counts()

                total_cls = len(subset)
                unique_vals = df[feature].unique()
                smoothed_probs = {
                    val: (val_counts.get(val, 0) + 1) / (total_cls + len(unique_vals))
                    for val in unique_vals
                }
                self.feature_probs[feature][cls] = smoothed_probs


