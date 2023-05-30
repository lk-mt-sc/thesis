import pickle


class Run:
    def __init__(
            self,
            id_,
            path,
            data,
            features,
            bboxes,
            bboxes_bottomup,
            ious,
            detection_scores,
            pose_estimation_scores,
            metrics):
        self.id = id_
        self.path = path
        self.data = data
        self.features = features
        self.bboxes = bboxes
        self.bboxes_bottomup = bboxes_bottomup
        self.ious = ious
        self.detection_scores = detection_scores
        self.pose_estimation_scores = pose_estimation_scores
        self.metrics = metrics

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
