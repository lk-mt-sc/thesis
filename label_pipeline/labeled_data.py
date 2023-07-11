import pickle


class LabeledData():
    def __init__(self, n_labels, feature_x, feature_y, scores):
        assert len(feature_x) == n_labels
        assert len(feature_y) == n_labels
        self.n_labels = n_labels
        self.labels = [(-1, 0, False, False)] * self.n_labels
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.scores = scores

    def add_labels(self, step, labels):
        self.labels[step] = labels

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
