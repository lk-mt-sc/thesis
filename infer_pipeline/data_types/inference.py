import os
import imp
import glob
import json
import time
import shutil
import subprocess
from statistics import mean
from datetime import datetime

import torch
from torchvision.ops import box_iou

from data_types.run import Run
from data_types.feature import Feature
from utils import collect_image_infos, cvt_to_coco_json
from manager.dataset_manager import InterpolationKeypoints
from common import MMPOSE_DIR, MMPOSE_TEST_SCRIPT, MMPOSE_DATASET_DIR
from common import MMDETECTION_DIR, MMDETECTION_TEST_SCRIPT
from common import INFERENCES_DIR


class Inference:
    def __init__(self, metadata):
        self.id = metadata['id']
        self.name = metadata['name']
        self.datetime_timestamp = metadata['datetime'] if 'datetime' in metadata else None
        if self.datetime_timestamp is None:
            self.datetime = None
        else:
            self.datetime = datetime.fromtimestamp(self.datetime_timestamp).strftime('%d.%m.%Y %H:%M:%S')
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

    def infer(self, inference_progress, existing_dataset, dataset_type):
        self.datetime_timestamp = datetime.timestamp(datetime.now())

        out_dir = os.path.join(INFERENCES_DIR, self.id)
        os.mkdir(out_dir)

        mmdetection_result_dir = os.path.join(out_dir, 'mmdetection_result_dir')
        os.mkdir(mmdetection_result_dir)

        mmdetection_work_dir = os.path.join(out_dir, 'mmdetection_work_dir')
        os.mkdir(mmdetection_work_dir)

        mmdetection_config = self.mmdetection_model.config
        mmdetection_checkpoint = self.mmdetection_model.checkpoint

        mmpose_config = self.mmpose_model.config
        mmpose_checkpoint = self.mmpose_model.checkpoint
        config_name = mmpose_config.split('/')[-1].split('.')[0]
        config = imp.load_source(config_name, mmpose_config)

        data_mode = config.data_mode
        assert data_mode == 'topdown' or data_mode == 'bottomup'

        if existing_dataset is None:
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
        else:
            dataset_dir = existing_dataset
            ann_file = os.path.join(dataset_dir, 'ann_file.json')

        if data_mode == 'topdown':
            persistent_detection_found = False
            if existing_dataset is not None:
                dataset_properties = existing_dataset.split('/')[-1]
                dataset_properties = dataset_properties.replace('dataset_', '')
                config_name = mmdetection_config.split('/')[-1].split('.')[0]
                persistent_result_dir = os.path.join(INFERENCES_DIR, 'persistent_detections',
                                                     dataset_properties + '_' + config_name, 'mmdetection_result_dir')
                persistent_results_pickle_file_path = os.path.join(persistent_result_dir, 'persistent.results.pkl')
                persistent_results_json_file_path = os.path.join(persistent_result_dir, 'persistent.bbox.json')

                if os.path.exists(persistent_results_pickle_file_path) and \
                        os.path.exists(persistent_results_json_file_path):
                    persistent_detection_found = True
                    results_pickle_file_path = os.path.join(mmdetection_result_dir, str(self.id) + '.results.pkl')
                    results_json_file_path = os.path.join(mmdetection_result_dir, str(self.id) + '.bbox.json')
                    shutil.copyfile(persistent_results_pickle_file_path, results_pickle_file_path)
                    shutil.copyfile(persistent_results_json_file_path, results_json_file_path)
                    ann_file = os.path.join(existing_dataset, 'ann_file.json')

            if not persistent_detection_found:
                mmdetection_outfile_prefix = os.path.join(mmdetection_result_dir, str(self.id))
                mmdetection_result_dump_file = os.path.join(mmdetection_result_dir,  str(self.id) + '.results.pkl')

                inference_progress.value = 'BB. DETECTION STARTUP'

                mmdetection_args = [
                    'python',
                    MMDETECTION_TEST_SCRIPT,
                    mmdetection_config,
                    mmdetection_checkpoint,
                    '--work-dir',
                    mmdetection_work_dir,
                    '--out',
                    mmdetection_result_dump_file,
                    '--cfg-options',
                    f'test_dataloader.dataset.ann_file={ann_file}',
                    f'test_evaluator.ann_file={ann_file}',
                    f'test_evaluator.outfile_prefix={mmdetection_outfile_prefix}'
                ]

                start = time.time()

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

                end = time.time()
                duration = end-start
                print('Duration of bounding box detection on the inference data (tot./avg. run/avg. image):')
                print(f'{str(round(duration / 60, 2))} min/{str(round(duration / 64, 2))} sec/{str(round(duration / 12372, 4))} sec')
                results_pickle_file_path = mmdetection_result_dump_file
                results_json_file_path = mmdetection_outfile_prefix + '.bbox.json'

                inference_progress.value = 'REMOVING LOW SCORES'

                with open(results_json_file_path, 'r', encoding='utf8') as results_file:
                    results = json.load(results_file)

                    image_ids = []
                    for result in results:
                        image_ids.append(result['image_id'])

                    ids = set()
                    duplicate_ids = []
                    for id_ in image_ids:
                        if id_ in ids:
                            duplicate_ids.append(id_)
                        else:
                            ids.add(id_)

                    for id_ in duplicate_ids:
                        duplicates = []
                        for result in results:
                            if result['image_id'] == id_:
                                duplicates.append(result)
                        duplicates = sorted(duplicates, key=lambda d: d['score'], reverse=True)
                        for duplicate in duplicates[1:]:
                            results.remove(duplicate)

                with open(results_json_file_path, 'w', encoding='utf8') as results_file:
                    json.dump(results, results_file)

                inference_progress.value = 'CALC. AVG. CONFIDENCE'

                scores = []
                with open(results_json_file_path, 'r', encoding='utf8') as results_file:
                    results = json.load(results_file)
                    assert len(results) == 12372

                    for result in results:
                        scores.append(result['score'])

                print(f'Average confidence of bounding box detection on the inference data: {mean(scores)}')

        mmpose_result_dir = os.path.join(out_dir, 'mmpose_result_dir')
        os.mkdir(mmpose_result_dir)

        mmpose_work_dir = os.path.join(out_dir, 'mmpose_work_dir')
        os.mkdir(mmpose_work_dir)

        mmpose_outfile_prefix = os.path.join(mmpose_result_dir, str(self.id))

        inference_progress.value = 'POSE EST. STARTUP'

        mmpose_args = [
            'python',
            MMPOSE_TEST_SCRIPT,
            mmpose_config,
            mmpose_checkpoint,
            '--work-dir',
            mmpose_work_dir,
            '--cfg-options',
            f'test_dataloader.dataset.ann_file={ann_file}',
            f'test_evaluator.ann_file={ann_file}',
            f'test_evaluator.outfile_prefix={mmpose_outfile_prefix}'
        ]

        if data_mode == 'topdown':
            bbox_file_path = os.path.join(mmpose_work_dir, 'bbox_file.json')
            shutil.copyfile(results_json_file_path, bbox_file_path)
            mmpose_args.append(f'test_dataloader.dataset.bbox_file={bbox_file_path}')

        start = time.time()

        pose_estimation = subprocess.Popen(
            mmpose_args,
            cwd=MMPOSE_DIR,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )

        while True:
            line = pose_estimation.stdout.readline()
            if not line:
                print()
                break
            line = line.rstrip()
            if 'mmengine - INFO - Epoch(test)' in line:
                tracker = line[line.find('[') + 1: line.find(']')].split('/')
                progress_percentage = int(int(tracker[0])/int(tracker[1]) * 100)
                inference_progress.value = 'POSE EST. ' + f'{progress_percentage}%'

        end = time.time()
        duration = end-start
        print('Duration of pose estimation on the inference data (tot./avg. run/avg. image):')
        print(str(round(duration / 60, 2)), str(round(duration / 64, 2)), str(round(duration / 12372, 4)))
        results_json_file_path = mmpose_outfile_prefix + '.keypoints.json'

        with open(ann_file, 'r') as annotations_file:
            annotations = json.load(annotations_file)

        if data_mode == 'bottomup':
            bbox_file_path = os.path.join(INFERENCES_DIR, 'bbox_bottomup.json')

        with open(bbox_file_path, 'r') as bbox_file:
            pred_bboxes = json.load(bbox_file)

        with open(results_json_file_path, 'r') as results_file:
            results = json.load(results_file)

        n_data = len(self.data)
        for i, data in enumerate(self.data):

            progress_percentage = int(((i + 1)/n_data) * 100)
            inference_progress.value = 'SAVING INFER RES. ' + f'{progress_percentage}%'

            images = data.get_images()
            bboxes = []
            detection_scores = []
            pose_estimation_scores = []
            features = []
            dataset_keypoints = dataset_type.keypoints
            n_keypoints = len(dataset_keypoints)
            for keypoint in dataset_keypoints:
                features.append(Feature(keypoint))
            for image in images:
                image_filename = str(data.id) + '_' + image.split('/')[-1]
                for dataset_image in annotations['images']:
                    if dataset_image['file_name'].split('/')[-1] == image_filename:
                        image_id = dataset_image['id']

                pred_bbox = None
                for bbox in pred_bboxes:
                    if bbox['image_id'] == image_id:
                        pred_bbox = bbox

                if data_mode == 'topdown':
                    result = next(result for result in results if result['image_id'] == image_id)
                    keypoints = result['keypoints']
                    x_coord = keypoints[0:: 3]
                    y_coord = keypoints[1:: 3]
                    for i, (x, y) in enumerate(zip(x_coord, y_coord)):
                        features[i].add(x, y)

                if data_mode == 'bottomup':
                    x = int(pred_bbox['bbox'][0])
                    y = int(pred_bbox['bbox'][1])
                    w = int(pred_bbox['bbox'][2])
                    h = int(pred_bbox['bbox'][3])
                    pred_bbox_xyxy = [x, y, x + w, y + h]
                    pred_bbox_tensor = torch.FloatTensor(pred_bbox_xyxy)
                    pred_bbox_tensor = pred_bbox_tensor.unsqueeze(0)

                    poses = []
                    for result in results:
                        if result['image_id'] == image_id:
                            poses.append(result)

                    preds = []
                    for pose in poses:
                        keypoints = pose['keypoints']
                        x_coord = keypoints[0::3]
                        y_coord = keypoints[1::3]

                        bbox = [int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))]
                        bbox_tensor = torch.FloatTensor(bbox)
                        bbox_tensor = bbox_tensor.unsqueeze(0)

                        iou = box_iou(pred_bbox_tensor, bbox_tensor)
                        score = pose['score']
                        if iou > 0.3:
                            preds.append({
                                'iou': iou,
                                'bbox': bbox,
                                'x_coord': x_coord,
                                'y_coord': y_coord,
                                'score': score
                            })

                    preds = sorted(preds, key=lambda d: d['score'], reverse=True)

                    if not preds:
                        preds.append({
                            'x_coord': [-1 for i in range(n_keypoints)],
                            'y_coord': [-1 for i in range(n_keypoints)],
                            'score': -1
                        })

                    result = preds[0]
                    x_coord = result['x_coord']
                    y_coord = result['y_coord']
                    for i, (x, y) in enumerate(zip(x_coord, y_coord)):
                        features[i].add(x, y)

                bboxes.append(pred_bbox['bbox'])
                detection_scores.append(pred_bbox['score'])
                pose_estimation_scores.append(result['score'])

            left_shoulder = next(f for f in features if f.name == InterpolationKeypoints.LEFT_SHOULDER.value)
            right_shoulder = next(f for f in features if f.name == InterpolationKeypoints.RIGHT_SHOULDER.value)
            neck = next(f for f in features if f.name == InterpolationKeypoints.NECK.value)

            left_ear = next(f for f in features if f.name == InterpolationKeypoints.LEFT_EAR.value)
            right_ear = next(f for f in features if f.name == InterpolationKeypoints.RIGHT_EAR.value)
            head = next(f for f in features if f.name == InterpolationKeypoints.HEAD.value)

            self.interpolate_keypoint(neck, left_shoulder, right_shoulder)
            self.interpolate_keypoint(head, left_ear, right_ear)

            run = Run(data.id, data, features, bboxes, detection_scores, pose_estimation_scores)
            run.save(os.path.join(out_dir, f'run_{data.id}.pkl'))

        self.store_metadata(out_dir)
        self.load_runs()

    def interpolate_keypoint(self, target, source_1, source_2):
        assert len(source_1.x) == len(source_2.x)
        assert len(source_1.y) == len(source_2.y)
        for s1_x, s1_y, s2_x, s2_y in zip(source_1.x, source_1.y, source_2.x, source_2.y):
            if -1 in (s1_x, s1_y, s2_x, s2_y):
                continue
            x1 = min(s1_x, s2_x)
            y1 = min(s1_y, s2_y)
            x2 = max(s1_x, s2_x)
            y2 = max(s1_y, s2_y)
            target.add(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

    def store_metadata(self, out_dir):
        metadata_file = open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf8')
        metadata = {
            'id': self.id,
            'name': self.name,
            'datetime': self.datetime_timestamp,
            'mmpose_model': str(self.mmpose_model),
            'mmpose_model_config': self.mmpose_model.config,
            'mmpose_model_checkpoint': self.mmpose_model.checkpoint,
            'mmdetection_model': str(self.mmdetection_model),
            'mmdetection_model_config': self.mmdetection_model.config,
            'mmdetection_model_checkpoint': self.mmdetection_model.checkpoint,
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
