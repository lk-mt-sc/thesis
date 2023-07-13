import pickle

import numpy as np


class LabeledData():
    def __init__(self, feature_x, feature_y, candidates_x, candidates_y):
        assert len(feature_x.steps) == len(feature_y.steps)
        assert feature_x.scores == feature_y.scores
        self.n_labels = len(feature_x.steps)
        # -1 = not labeled, 0 = correct, 1 = incorrect, 2 = incorrect, hidden
        # bools: bb cuts climber, bb cuts climber (TV), bb cuts climber (dark), side swap, mark x, mark y,
        self.labels = [(-1, False, False, False, False, False, False)] * self.n_labels
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.candidates_x = candidates_x
        self.candidates_y = candidates_y
        self.scores = np.clip(feature_x.scores, 0.0, 1.0)
        self.title = ' '.join(word.capitalize() for word in feature_x.name.split('_'))

    def add_labels(self, step, labels):
        self.labels[step] = labels

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
