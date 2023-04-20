import pickle


class Run:
    def __init__(self, id, data):
        self.id = id
        self.data = data

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
