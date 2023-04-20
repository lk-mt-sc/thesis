import pickle


class Run:
    def __init__(self, id_, data, features):
        self.id = id_
        self.data = data
        self.features = features

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
