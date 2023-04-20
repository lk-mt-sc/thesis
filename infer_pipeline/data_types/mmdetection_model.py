class MMDetectionModel():
    def __init__(self, name, key_metric, checkpoint, config):
        self.name = name
        self.key_metric = key_metric
        self.checkpoint = checkpoint
        self.config = config

    def __str__(self):
        return f'{self.name} | {self.key_metric}'

    def __eq__(self, other):
        equal = self.name == other.name and self.key_metric == other.key_metric
        return equal
