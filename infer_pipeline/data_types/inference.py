import os
import glob
import json
from datetime import datetime

from my_mmpose.apis import MyMMPoseInferencer

from data_types.run import Run
from utils import Suppressor
from common import MMPOSE_DIR, MMPOSE_CHECKPOINTS_DIR
from common import MMDETECTION_DIR, MMDETECTION_CHECKPOINTS_DIR
from common import INFERENCES_DIR


class Inference:
    def __init__(self, metadata):
        self.id = metadata['id']
        self.name = metadata['name']
        self.datetime_timestamp = None
        self.datetime = None
        self.mmpose_model = metadata['mmpose_model']
        self.mmdetection_model = metadata['mmdetection_model']
        self.data = metadata['data']
        self.duration = metadata['duration']
        self.description = metadata['description']
        self.runs = []

        self.load_runs()

    def __str__(self):
        print_str = f'{self.name}'
        if self.datetime is not None:
            print_str += f' | {self.datetime} | {self.id}'
        else:
            print_str += f' | {self.id}'
        return print_str

    def infer(self, inference_progress):
        self.datetime_timestamp = datetime.timestamp(datetime.now())
        self.datetime = datetime.fromtimestamp(self.datetime_timestamp).strftime('%d.%m.%Y %H:%M:%S')

        out_dir = os.path.join(INFERENCES_DIR, self.id)
        os.mkdir(out_dir)

        with Suppressor():
            inferencer = MyMMPoseInferencer(
                pose2d=os.path.join(MMPOSE_DIR, self.mmpose_model.config),
                pose2d_weights=os.path.join(MMPOSE_CHECKPOINTS_DIR, self.mmpose_model.checkpoint),
                det_model=os.path.join(MMDETECTION_DIR, self.mmdetection_model.config),
                det_weights=os.path.join(MMDETECTION_CHECKPOINTS_DIR, self.mmdetection_model.checkpoint),
                det_cat_ids=[0])

        n_total_images = 0
        for data in self.data:
            images = data.get_images()
            n_total_images += len(images)

        n_processed_images = 0
        for data in self.data:
            images = data.get_images()

            features = []
            for image in images:
                with Suppressor():
                    result_generator = inferencer(image)
                    result = next(result_generator)
                    # TODO: fill features with result data
                n_processed_images += 1
                inference_progress.value = int(n_processed_images/n_total_images * 100)

            run = Run(data.id, data, features)
            run.save(os.path.join(out_dir, f'run_{data.id}.pkl'))

        self.store_metadata(out_dir)
        self.load_runs()

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
