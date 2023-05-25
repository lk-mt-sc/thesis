import pickle


class Run:
    def __init__(self, id_, data, features, bboxes, detection_scores, pose_estimation_scores):
        self.id = id_
        self.data = data
        self.features = features
        self.bboxes = bboxes
        self.detection_scores = detection_scores
        self.pose_estimation_scores = pose_estimation_scores

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
