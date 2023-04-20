import os
import glob
import json
from datetime import datetime

from mmpose.apis import MMPoseInferencer

from data_types.run import Run
from utils import Suppressor
from common import MMPOSE_DIR, INFERENCES_DIR, MMPOSE_CHECKPOINTS_DIR


class Inference:
    def __init__(self, metadata):
        self.id = metadata['id']
        self.name = metadata['name']
        self.datetime_timestamp = metadata['datetime']
        self.datetime = datetime.fromtimestamp(self.datetime_timestamp).strftime('%d.%m.%Y %H:%M:%S')
        self.mmpose_model = metadata['mmpose_model']
        self.mmdetection_model = metadata['mmdetection_model']
        self.data = metadata['data']
        self.duration = metadata['duration']
        self.description = metadata['description']
        self.runs = []

        self.load_runs()

    def __str__(self):
        return f'{self.name} | {self.datetime} | {self.id}'

    def infer(self):
        pass

    def store_metadata(self, out_dir):
        metadata_file = open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf8')
        metadata = {
            'id': self.id,
            'name': self.name,
            'datetime': self.datetime_timestamp,
            'mmpose_model': str(self.mmpose_model),
            'mmdetection_model': str(self.mmdetection_model),
            'data': [int(d.id) for d in self.data],
            'duration': self.duration,
            'description': self.description
        }
        json.dump(metadata, metadata_file)

    def load_runs(self):
        runs = sorted(glob.glob(os.path.join(INFERENCES_DIR, self.id, '*.pkl')))
        if not runs:
            self.runs = []
        else:
            self.runs.clear()
            for run in runs:
                self.runs.append(Run.load(run))

    def get_run(self, id_):
        for run in self.runs:
            if run.id == id_:
                return run
        return None
