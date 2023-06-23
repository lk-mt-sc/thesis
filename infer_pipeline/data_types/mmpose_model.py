class MMPoseModel():
    def __init__(
            self,
            section,
            arch,
            dataset,
            input_size,
            key_metric_value,
            key_metric_name,
            checkpoint,
            config,
            transfer_learned=False,
            multi_frame_mmpose029=False
    ):
        self.section = section
        self.arch = arch
        self.dataset = dataset
        self.input_size = input_size
        self.key_metric_value = key_metric_value
        self.key_metric_name = key_metric_name
        self.checkpoint = checkpoint
        self.config = config
        self.transfer_learned = transfer_learned
        self.multi_frame_mmpose029 = multi_frame_mmpose029

    @classmethod
    def get_from_selection_string(cls, selection_str):
        selection_str_split = selection_str.split(' | ')
        return MMPoseModel(
            section=selection_str_split[5],
            arch=selection_str_split[6],
            dataset=selection_str_split[1],
            input_size=selection_str_split[4],
            key_metric_value=float(selection_str_split[2]),
            key_metric_name=selection_str_split[3],
            checkpoint=None,
            config=None,
            transfer_learned=selection_str_split[0] == 'TL: True'
        )

    def __str__(self):
        return f'TL: {str(self.transfer_learned)} | {self.dataset} | {self.key_metric_value} | {self.key_metric_name} | {self.input_size} | {self.section} | {self.arch}'

    def __eq__(self, other):
        equal = self.section == other.section \
            and self.arch == other.arch \
            and self.dataset == other.dataset \
            and self.input_size == other.input_size \
            and self.key_metric_value == other.key_metric_value \
            and self.key_metric_name == other.key_metric_name \
            and self.transfer_learned == other.transfer_learned
        return equal
