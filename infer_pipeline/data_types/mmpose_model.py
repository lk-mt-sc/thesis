class MMPoseModel():
    def __init__(self, section, arch, dataset, input_size, key_metric, checkpoint, config):
        self.section = section
        self.arch = arch
        self.dataset = dataset
        self.input_size = input_size
        self.key_metric = key_metric
        self.checkpoint = checkpoint
        self.config = config

    def __str__(self):
        return f'{self.dataset.value.upper()} | {self.key_metric} | {self.input_size} | {self.section} | {self.arch}'

    def __eq__(self, other):
        equal = self.section == other.section \
            and self.arch == other.arch \
            and self.dataset == other.dataset \
            and self.input_size == other.input_size \
            and self.key_metric == other.key_metric
        return equal
