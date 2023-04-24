import os
import glob
import json
import shutil
import subprocess
from datetime import datetime

from data_types.run import Run
from utils import collect_image_infos, cvt_to_coco_json
from common import MMPOSE_DIR, MMPOSE_CHECKPOINTS_DIR, MMPOSE_TEST_SCRIPT, MMPOSE_DATASET_DIR, MMPOSE_RUNS_DIR
from common import MMDETECTION_DIR, MMDETECTION_CHECKPOINTS_DIR, MMDETECTION_TEST_SCRIPT
from common import WORKING_DIR, INFERENCES_DIR


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

        mmdetection_config = os.path.join(MMDETECTION_DIR, self.mmdetection_model.config)
        mmdetection_checkpoint = os.path.join(MMDETECTION_CHECKPOINTS_DIR, self.mmdetection_model.checkpoint)

        temp_dir = os.path.join(out_dir, 'temp')
        mmdetection_work_dir = os.path.join(temp_dir, 'work_dir')
        os.makedirs(mmdetection_work_dir)

        dataset_dir = MMPOSE_DATASET_DIR + f'_{self.id}'
        os.mkdir(dataset_dir)
        for i, data in enumerate(self.data):
            inference_progress.value = f'DATA PREP. {i}/{len(self.data)}'
            images = data.get_images()
            for image in images:
                src = image
                dst = os.path.join(dataset_dir, f"{data.id}_{image.split('/')[-1]}")
                shutil.copyfile(src, dst)

        inference_progress.value = 'ANN. FILE CREATION'
        image_infos = collect_image_infos(dataset_dir)
        image_list_coco_format = cvt_to_coco_json(image_infos)
        ann_file = os.path.join(dataset_dir, 'ann_file.json')

        with open(ann_file, 'w', encoding='utf8') as file:
            json.dump(image_list_coco_format, file)

        mmdetection_outfile_prefix = os.path.join(out_dir, 'temp', 'result_dir', str(self.id))
        mmdetection_num_workers = 2
        mmdetection_batch_size = 2

        inference_progress.value = 'BB. DETECTION STARTUP'

        mmdetection_args = [
            'python',
            MMDETECTION_TEST_SCRIPT,
            mmdetection_config,
            mmdetection_checkpoint,
            '--work-dir',
            mmdetection_work_dir,
            '--cfg-options',
            f'test_dataloader.dataset.ann_file={ann_file}',
            f'test_dataloader.batch_size={mmdetection_batch_size}',
            f'test_dataloader.num_workers={mmdetection_num_workers}',
            f'test_evaluator.outfile_prefix={mmdetection_outfile_prefix}',
            f'test_evaluator.ann_file={ann_file}',
        ]

        detection = subprocess.Popen(
            mmdetection_args,
            cwd=MMDETECTION_DIR,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )

        while True:
            line = detection.stdout.readline()
            if not line:
                print()
                break
            line = line.rstrip()
            if 'mmengine - INFO - Epoch(test)' in line:
                tracker = line[line.find('[') + 1: line.find(']')].split('/')
                progress_percentage = int(int(tracker[0])/int(tracker[1]) * 100)
                inference_progress.value = 'BB. DETECTION ' + f'{progress_percentage}%'

        shutil.rmtree(temp_dir)
        shutil.rmtree(dataset_dir)

        # run = Run(data.id, data, features)
        # run.save(os.path.join(out_dir, f'run_{data.id}.pkl'))

        # self.store_metadata(out_dir)
        # self.load_runs()

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
