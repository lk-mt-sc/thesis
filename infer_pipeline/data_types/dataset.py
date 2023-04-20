class Dataset():
    def __init__(self, name, keypoints, skeleton):
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton

    def __str__(self):
        return self.name
