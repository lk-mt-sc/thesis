class MMDetectionModel():
    def __init__(self, name, key_metric_value, key_metric_name, checkpoint, config):
        self.name = name
        self.key_metric_value = key_metric_value
        self.key_metric_name = key_metric_name
        self.checkpoint = checkpoint
        self.config = config

    @classmethod
    def get_from_selection_string(cls, selection_str):
        selection_str_split = selection_str.split(' | ')
        return MMDetectionModel(
            name=selection_str_split[0],
            key_metric_value=float(selection_str_split[1]),
            key_metric_name=selection_str_split[2],
            checkpoint=None,
            config=None
        )

    def __str__(self):
        return f'{self.name} | {self.key_metric_value} | {self.key_metric_name}'

    def __eq__(self, other):
        equal = self.name == other.name \
            and self.key_metric_value == other.key_metric_value \
            and self.key_metric_name == other.key_metric_name
        return equal
